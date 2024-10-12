import sys
sys.path.insert(0, '/home/myid/krb84578/ViTST/code')
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import collections.abc

import torch
from transformers import *
from sklearn.metrics import *

from torchvision.transforms import (
    Compose,
    ToTensor,
)

from models.vision_text_dual_encoder.modeling_vision_text_dual_encoder import VisionTextDualEncoderModelForClassification
from models.vision_text_dual_encoder.configuration_vision_text_dual_encoder import VisionTextDualEncoderForClassificationConfig

from transformers import (
    ViTConfig, 
    BertConfig, 
    ViTFeatureExtractor,
    BertTokenizer,
    AutoFeatureExtractor,
    AutoTokenizer
)

from datasets import load_dataset, load_metric, Dataset, Image

from load_data import get_data_split 

def one_hot(y_):
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def load_or_initialize_model(image_model_path, text_model_path, output_dir, image_size, grid_layout, num_classes, continue_training):
    if os.path.exists(output_dir) and continue_training:
        checkpoints = [d for d in os.listdir(output_dir) if "checkpoint" in d]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            ft_model_path = os.path.join(output_dir, latest_checkpoint)
            config = VisionTextDualEncoderForClassificationConfig.from_pretrained(ft_model_path)
            return VisionTextDualEncoderModelForClassification.from_pretrained(ft_model_path, config=config)
    return VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
        image_model_path, text_model_path, num_classes=num_classes, 
        vision_image_size=image_size, vision_grid_layout=grid_layout
    )

def compute_metrics(eval_pred, num_classes, is_multilabel=False):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    metric = load_metric("accuracy")
    accuracy = metric.compute(predictions=preds, references=labels)["accuracy"]

    if num_classes == 2:
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        denoms = np.sum(np.exp(predictions), axis=1).reshape((-1, 1))
        probs = np.exp(predictions) / denoms
        auc = roc_auc_score(labels, probs[:, 1])
        aupr = average_precision_score(labels, probs[:, 1])
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auroc": auc, "auprc": aupr}

    elif num_classes > 2:
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        f1 = f1_score(labels, preds, average="macro")
        auc = roc_auc_score(one_hot(labels), predictions) if is_multilabel else 0
        aupr = average_precision_score(one_hot(labels), predictions) if is_multilabel else 0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auroc": auc, "auprc": aupr}

    else:
        rmse = mean_squared_error(labels, predictions, squared=False)
        mape = mean_absolute_percentage_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        return {"rmse": rmse, "mape": mape, "mae": mae}


def preprocess(example_batch, transforms, tokenizer, max_length):
    example_batch["pixel_values"] = [transforms(image.convert("RGB")) for image in example_batch["image"]]
    text_embeddings = tokenizer([text for text in example_batch["text"]], 
                                padding='max_length', 
                                max_length=max_length, 
                                return_tensors="pt")
    example_batch["input_ids"] = text_embeddings["input_ids"]
    example_batch["attention_mask"] = text_embeddings["attention_mask"]
    return example_batch

def fine_tune_hf(
    image_model_path,
    text_model_path,
    freeze_vision_model,
    freeze_text_model,
    output_dir,
    train_dataset,
    val_dataset,
    test_dataset,
    image_size,
    grid_layout,
    num_classes,
    max_length,
    epochs,
    train_batch_size,
    eval_batch_size,
    save_steps,
    logging_steps,
    learning_rate,
    seed,
    save_total_limit,
    do_train,
    continue_training
    ):  

    # loading model and feature extractor
    model = load_or_initialize_model(image_model_path, text_model_path, output_dir, image_size, grid_layout, num_classes, continue_training)

    # whether to freeze models
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    if freeze_vision_model == "True":
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = False
            print("freezed vision model parameter: ", name)
    if freeze_text_model == "True":
        for name, param in model.text_model.named_parameters():
            param.requires_grad = False
            print("freezed text model parameter: ", name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_path)
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    train_transforms = Compose([ToTensor()])
    val_transforms = Compose([ToTensor()])

    train_dataset.set_transform(lambda x: preprocess(x, train_transforms, tokenizer, max_length))
    val_dataset.set_transform(lambda x: preprocess(x, val_transforms, tokenizer, max_length))
    test_dataset.set_transform(lambda x: preprocess(x, val_transforms, tokenizer, max_length))

    if num_classes == 1:
        compute_metrics_fn = lambda x: compute_metrics(x, num_classes)
        best_metric = "rmse"
    elif num_classes == 2:
        compute_metrics_fn = lambda x: compute_metrics(x, num_classes)
        best_metric = "auroc"
    else:
        compute_metrics_fn = lambda x: compute_metrics(x, num_classes, is_multilabel=True)
        best_metric = "accuracy"

    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "logging_dir": os.path.join(output_dir, "runs/"),
        "save_total_limit": save_total_limit,
        "seed": seed,
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
        "metric_for_best_model": best_metric
    }

    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=lambda x: {"pixel_values": torch.stack([example["pixel_values"] for example in x]), 
                                 "input_ids": torch.stack([example["input_ids"] for example in x]), 
                                 "attention_mask": torch.stack([example["attention_mask"] for example in x]), 
                                 "labels": torch.tensor([example["label"] for example in x])},
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    if do_train:
        trainer.train()

    predictions = trainer.predict(test_dataset)
    logits, labels = predictions.predictions, predictions.label_ids
    ypred = np.argmax(logits, axis=1)
    denoms = np.sum(np.exp(logits), axis=1).reshape((-1, 1))
    probs = np.exp(logits) / denoms

    if num_classes == 1:
        rmse = mean_squared_error(labels, logits, squared=False)
        mape = mean_absolute_percentage_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        return 0., 0., 0., 0., 0., 0., rmse, mape, mae
    elif num_classes == 2:
        acc = np.sum(labels.ravel() == ypred.ravel()) / labels.shape[0]
        precision = precision_score(labels, ypred)
        recall = recall_score(labels, ypred)
        F1 = f1_score(labels, ypred)
        auc = roc_auc_score(labels, probs[:, 1])
        aupr = average_precision_score(labels, probs[:, 1])
        return acc, precision, recall, F1, auc, aupr, 0., 0., 0.
    else:
        acc = np.sum(labels.ravel() == ypred.ravel()) / labels.shape[0]
        precision = precision_score(labels, ypred, average="macro")
        recall = recall_score(labels, ypred, average="macro") 
        F1 = f1_score(labels, ypred, average="macro")
        auc = roc_auc_score(one_hot(labels), probs)
        aupr = average_precision_score(one_hot(labels), probs)
        return acc, precision, recall, F1, auc, aupr, 0., 0., 0.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='P12')    
    parser.add_argument('--dataset_prefix', type=str, default='') 
    parser.add_argument('--withmissingratio', default=False)
    parser.add_argument('--feature_removal_level', type=str, default='no_removal')

    # arguments for huggingface training
    parser.add_argument('--image_model', type=str, default='vit') 
    parser.add_argument('--image_model_path', type=str, default=None)
    parser.add_argument('--text_model', type=str, default='bert') 
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=36) 
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1799)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--upsample', default=False)

    parser.add_argument('--grid_layout', default=None)
    parser.add_argument('--image_size', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--freeze_vision_model', type=str, default="False")
    parser.add_argument('--freeze_text_model', type=str, default="False")
    parser.add_argument('--continue_training', action='store_true')

    args = parser.parse_args()

    dataset = args.dataset
    dataset_prefix = args.dataset_prefix
    print(f'Dataset used: {dataset}, prefix: {dataset_prefix}.')

    upsample = args.upsample
    epochs = args.epochs
    image_size = grid_layout = None
    freeze_vision_model = args.freeze_vision_model
    freeze_text_model = args.freeze_text_model

    base_path = f'../../dataset/{dataset}data'
    num_classes = 2  # Assuming a binary classification task; modify accordingly.
    
    for k in range(args.n_splits):
        split_idx = k + 1
        print(f'Split id: {split_idx}')
        split_path = f'/splits/{dataset}_split{split_idx}.npy'

        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_idx, dataset=dataset, prefix=dataset_prefix, upsample=upsample)
        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        for m in range(args.n_runs):
            print(f'- - Run {m + 1} - -')
            acc, precision, recall, F1, auc, aupr, rmse, mape, mae = fine_tune_hf(
                image_model_path=args.image_model_path,
                text_model_path=args.text_model_path,
                freeze_vision_model=freeze_vision_model,
                freeze_text_model=freeze_text_model,
                output_dir=args.output_dir,
                train_dataset=Ptrain,
                val_dataset=Pval,
                test_dataset=Ptest,
                image_size=image_size,
                grid_layout=grid_layout,
                num_classes=num_classes,
                max_length=args.max_length,
                epochs=epochs,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
                save_total_limit=args.save_total_limit,
                do_train=args.do_train,
                continue_training=args.continue_training
            )

            test_report = (f'Testing: Precision = {precision:.2f} | Recall = {recall:.2f} | F1 = {F1:.2f}\n'
                           f'Testing: AUROC = {auc:.2f} | AUPRC = {aupr:.2f} | Accuracy = {acc:.2f}\n'
                           f'Testing: RMSE = {rmse:.2f} | MAPE = {mape:.2f} | MAE = {mae:.2f}\n')

            print(test_report)
            result_path = "train_result.txt" if args.do_train else "test_result.txt"
            with open(os.path.join(args.output_dir, result_path), "w+") as f:
                f.write(test_report)




# Here’s a high-level overview of the modular changes you should focus on for optimization:

# ### 1. **Model Loading**
#    - Create a separate module or function to handle model initialization and checkpoint loading.
#    - Ensure that the model loading logic can handle both new training sessions and resuming from the latest checkpoint.

# ### 2. **Data Preprocessing**
#    - Modularize the preprocessing steps for datasets (training, validation, and testing). Create a single preprocessing function that can be reused for all datasets to reduce redundancy.

# ### 3. **Metrics Calculation**
#    - Move the metric calculations into a dedicated module or function, generalizing across different classification tasks (binary, multilabel, regression). This avoids repeating code and makes it easier to adjust metrics.

# ### 4. **Training and Checkpointing**
#    - Encapsulate the training and checkpointing logic in a separate function. This ensures that the model's state, optimizer, and learning rate scheduler are saved at each checkpoint, and they can be reloaded seamlessly.

# ### 5. **Inference and Testing**
#    - Separate the inference (test-time) logic from the training logic. Have a dedicated function to load the best checkpoint for inference without needing to retrain or reinitialize.

# ### 6. **Logging and Error Handling**
#    - Introduce structured logging and error handling to track the progress and catch any issues during model loading, training, or evaluation. This will help when debugging or running long experiments.

# These modular changes will make your code more scalable, easier to debug, and optimized for reuse in future projects.