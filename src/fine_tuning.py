# src/fine_tuning.py

import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def fine_tune_model(iteration, model_name, train_texts, train_labels, test_texts, test_labels, num_labels):
    """
    Fine-tune a pre-trained model for sequence classification.

    Args:
        iteration (int): Current iteration number.
        model_name (str): Name of the pre-trained model.
        train_texts (list): List of training document texts.
        train_labels (list): List of training labels.
        test_texts (list): List of testing document texts.
        test_labels (list): List of testing labels.
        num_labels (int): Number of unique labels.

    Returns:
        str: Path to the fine-tuned model.
    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Tokenize datasets
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")

    # Create PyTorch datasets
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ClassificationDataset(train_encodings, train_labels)
    eval_dataset = ClassificationDataset(test_encodings, test_labels)

    # Define evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Define training arguments
    output_dir = f'./results_iteration_{iteration}'
    logging_dir = f'./logs_iteration_{iteration}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir=logging_dir,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        report_to='none',  # Disable WandB reporting to prevent configuration errors
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    fine_tuned_model_path = f'./fine-tuned-model-iteration-{iteration}'
    model.save_pretrained(fine_tuned_model_path)
    tokenizer.save_pretrained(fine_tuned_model_path)

    return fine_tuned_model_path

        
