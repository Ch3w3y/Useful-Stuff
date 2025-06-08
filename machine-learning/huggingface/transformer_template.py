#!/usr/bin/env python3
"""
HuggingFace Transformers Template
=================================

A comprehensive template for working with HuggingFace transformers for various NLP tasks:
- Text Classification
- Named Entity Recognition
- Question Answering
- Text Generation
- Summarization
- Translation

Author: Data Science Team
Date: 2024
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    pipeline, logging
)
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings

# Configuration
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

class TransformerTemplate:
    """
    A comprehensive class for working with HuggingFace transformers
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 task: str = "classification",
                 device: str = None):
        """
        Initialize the transformer template
        
        Args:
            model_name: Name of the pre-trained model
            task: Type of task (classification, ner, qa, generation, summarization)
            device: Device to run on (auto-detected if None)
        """
        self.model_name = model_name
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._get_model_for_task()
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"Initialized {task} model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_model_for_task(self):
        """Get the appropriate model class for the task"""
        if self.task == "classification":
            return AutoModelForSequenceClassification.from_pretrained(self.model_name)
        elif self.task == "ner":
            return AutoModelForTokenClassification.from_pretrained(self.model_name)
        elif self.task == "qa":
            return AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        elif self.task == "generation":
            return AutoModelForCausalLM.from_pretrained(self.model_name)
        elif self.task == "summarization" or self.task == "translation":
            return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            return AutoModel.from_pretrained(self.model_name)
    
    def preprocess_text_classification(self, 
                                     texts: List[str], 
                                     labels: Optional[List[int]] = None,
                                     max_length: int = 512) -> Dataset:
        """
        Preprocess data for text classification
        
        Args:
            texts: List of input texts
            labels: List of labels (optional for inference)
            max_length: Maximum sequence length
            
        Returns:
            Processed dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        data_dict = {"text": texts}
        if labels is not None:
            data_dict["labels"] = labels
        
        dataset = Dataset.from_dict(data_dict)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def preprocess_token_classification(self,
                                      texts: List[str],
                                      labels: Optional[List[List[str]]] = None,
                                      label_to_id: Optional[Dict[str, int]] = None,
                                      max_length: int = 512) -> Dataset:
        """
        Preprocess data for token classification (NER)
        
        Args:
            texts: List of input texts (already tokenized)
            labels: List of label sequences
            label_to_id: Mapping from label names to IDs
            max_length: Maximum sequence length
            
        Returns:
            Processed dataset
        """
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding=True,
                max_length=max_length
            )
            
            if labels is not None:
                labels_aligned = []
                for i, label in enumerate(examples["ner_tags"]):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []
                    
                    for word_idx in word_ids:
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    
                    labels_aligned.append(label_ids)
                
                tokenized_inputs["labels"] = labels_aligned
            
            return tokenized_inputs
        
        # Prepare data
        tokens_list = [text.split() for text in texts]
        data_dict = {"tokens": tokens_list}
        
        if labels is not None and label_to_id is not None:
            labels_ids = [[label_to_id.get(label, 0) for label in label_seq] 
                         for label_seq in labels]
            data_dict["ner_tags"] = labels_ids
        
        dataset = Dataset.from_dict(data_dict)
        
        # Tokenize and align labels
        tokenized_dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True
        )
        
        return tokenized_dataset
    
    def train_model(self,
                   train_dataset: Dataset,
                   eval_dataset: Optional[Dataset] = None,
                   output_dir: str = "./results",
                   num_epochs: int = 3,
                   batch_size: int = 16,
                   learning_rate: float = 2e-5,
                   warmup_steps: int = 500,
                   weight_decay: float = 0.01,
                   logging_steps: int = 10,
                   eval_steps: int = 500,
                   save_steps: int = 500) -> Trainer:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory for saving model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            logging_steps: Steps between logging
            eval_steps: Steps between evaluations
            save_steps: Steps between saves
            
        Returns:
            Trained trainer object
        """
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps,
            save_steps=save_steps,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to=None,  # Disable wandb/tensorboard logging
            save_total_limit=2,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Compute metrics function
        def compute_metrics(eval_pred: EvalPrediction):
            predictions, labels = eval_pred
            
            if self.task == "classification":
                predictions = np.argmax(predictions, axis=1)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted'
                )
                accuracy = accuracy_score(labels, predictions)
                return {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            elif self.task == "ner":
                predictions = np.argmax(predictions, axis=2)
                
                # Remove ignored index (special tokens)
                true_predictions = [
                    [p for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                true_labels = [
                    [l for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
                
                # Flatten lists
                true_predictions = [item for sublist in true_predictions for item in sublist]
                true_labels = [item for sublist in true_labels for item in sublist]
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, true_predictions, average='weighted'
                )
                accuracy = accuracy_score(true_labels, true_predictions)
                
                return {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            else:
                return {}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def predict(self, texts: List[str], batch_size: int = 16) -> Dict[str, Any]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        self.model.eval()
        
        if self.task == "classification":
            return self._predict_classification(texts, batch_size)
        elif self.task == "ner":
            return self._predict_ner(texts, batch_size)
        elif self.task == "generation":
            return self._predict_generation(texts, batch_size)
        else:
            raise NotImplementedError(f"Prediction not implemented for task: {self.task}")
    
    def _predict_classification(self, texts: List[str], batch_size: int) -> Dict[str, Any]:
        """Predict for classification task"""
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'confidence': [max(probs) for probs in all_probabilities]
        }
    
    def _predict_ner(self, texts: List[str], batch_size: int) -> Dict[str, Any]:
        """Predict for NER task"""
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
                is_split_into_words=False
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            all_predictions.extend(predictions.cpu().numpy())
        
        return {'predictions': all_predictions}
    
    def _predict_generation(self, texts: List[str], batch_size: int, 
                          max_length: int = 100) -> Dict[str, Any]:
        """Predict for generation task"""
        generated_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return {'generated_texts': generated_texts}
    
    def create_pipeline(self, task_type: str = None) -> pipeline:
        """
        Create a HuggingFace pipeline for inference
        
        Args:
            task_type: Type of pipeline task
            
        Returns:
            HuggingFace pipeline
        """
        task_type = task_type or self.task
        
        return pipeline(
            task_type,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def plot_training_history(self, trainer: Trainer, save_path: str = None):
        """
        Plot training history
        
        Args:
            trainer: Trained trainer object
            save_path: Path to save the plot
        """
        if not trainer.state.log_history:
            print("No training history available")
            return
        
        # Extract metrics
        train_loss = []
        eval_loss = []
        eval_accuracy = []
        steps = []
        
        for log in trainer.state.log_history:
            if 'loss' in log:
                train_loss.append(log['loss'])
                steps.append(log['step'])
            if 'eval_loss' in log:
                eval_loss.append(log['eval_loss'])
            if 'eval_accuracy' in log:
                eval_accuracy.append(log['eval_accuracy'])
        
        # Create plots
        fig, axes = plt.subplots(1, 2 if eval_loss else 1, figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Training loss
        axes[0].plot(steps, train_loss, label='Training Loss')
        if eval_loss:
            eval_steps = steps[-len(eval_loss):]
            axes[0].plot(eval_steps, eval_loss, label='Validation Loss')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Validation accuracy
        if eval_accuracy and len(axes) > 1:
            eval_steps = steps[-len(eval_accuracy):]
            axes[1].plot(eval_steps, eval_accuracy, label='Validation Accuracy', color='green')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")
    
    def load_model(self, path: str):
        """Load model and tokenizer"""
        self.model = self._get_model_for_task().from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model and tokenizer loaded from {path}")

# Example Usage Functions
def example_text_classification():
    """Example: Sentiment Analysis"""
    print("=== Text Classification Example ===")
    
    # Sample data
    texts = [
        "I love this movie! It's fantastic.",
        "This film is terrible. I hate it.",
        "The movie was okay, nothing special.",
        "Amazing cinematography and great acting!",
        "Boring and predictable storyline."
    ]
    labels = [1, 0, 2, 1, 0]  # 0: negative, 1: positive, 2: neutral
    
    # Initialize model
    model = TransformerTemplate(
        model_name="distilbert-base-uncased",
        task="classification"
    )
    
    # Preprocess data
    dataset = model.preprocess_text_classification(texts, labels)
    
    # Train model
    trainer = model.train_model(
        train_dataset=dataset,
        num_epochs=2,
        batch_size=4,
        output_dir="./sentiment_model"
    )
    
    # Make predictions
    new_texts = ["This movie is incredible!", "I didn't like it at all."]
    predictions = model.predict(new_texts)
    
    print("\nPredictions:")
    for text, pred, conf in zip(new_texts, predictions['predictions'], predictions['confidence']):
        print(f"Text: {text}")
        print(f"Prediction: {pred}, Confidence: {conf:.3f}\n")

def example_ner():
    """Example: Named Entity Recognition"""
    print("=== Named Entity Recognition Example ===")
    
    # Sample data (would normally be larger)
    texts = [
        "John works at Google in California",
        "Apple was founded by Steve Jobs"
    ]
    
    # For NER, you would typically use a pre-trained model
    model = TransformerTemplate(
        model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
        task="ner"
    )
    
    # Create pipeline for inference
    ner_pipeline = model.create_pipeline("ner")
    
    # Make predictions
    for text in texts:
        entities = ner_pipeline(text)
        print(f"Text: {text}")
        print("Entities:")
        for entity in entities:
            print(f"  {entity['word']}: {entity['entity']} (confidence: {entity['score']:.3f})")
        print()

def example_text_generation():
    """Example: Text Generation"""
    print("=== Text Generation Example ===")
    
    # Initialize model
    model = TransformerTemplate(
        model_name="gpt2",
        task="generation"
    )
    
    # Create pipeline
    generator = model.create_pipeline("text-generation")
    
    # Generate text
    prompts = [
        "The future of artificial intelligence is",
        "In a world where data science rules"
    ]
    
    for prompt in prompts:
        generated = generator(prompt, max_length=100, num_return_sequences=1)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[0]['generated_text']}\n")

if __name__ == "__main__":
    print("HuggingFace Transformers Template")
    print("=" * 50)
    print()
    
    # Check GPU availability
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
    print()
    
    # Run examples (uncomment to run)
    # example_text_classification()
    # example_ner()
    # example_text_generation()
    
    print("Examples complete! Uncomment example functions to run them.")
    print("\nUsage:")
    print("1. Initialize TransformerTemplate with your model and task")
    print("2. Preprocess your data using the appropriate method")
    print("3. Train the model using train_model()")
    print("4. Make predictions using predict() or create_pipeline()")
    print("5. Save your trained model using save_model()") 