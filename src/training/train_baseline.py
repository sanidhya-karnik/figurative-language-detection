import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import BaselineClassifier, DeBERTaClassifier, BertClassifier


class WeightedTrainer(Trainer):
    """
    Custom Trainer that uses class weights for imbalanced datasets
    """
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with class weights
        
        Args:
            model: The model
            inputs: The inputs dict
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted cross entropy loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss


class BaselineTrainer:
    """Train baseline models for Experiment 1"""
    
    def __init__(self, model_type='roberta', task='metaphor'):
        """
        Args:
            model_type: 'roberta', 'deberta', or 'bert'
            task: 'metaphor' or 'irony'
        """
        self.task = task
        
        if model_type == 'roberta':
            classifier = BaselineClassifier(
                model_name='roberta-base',
                num_labels=2,
                task_name=task
            )
        elif model_type == 'deberta':
            classifier = DeBERTaClassifier(
                model_name='microsoft/deberta-v3-base',
                num_labels=2,
                task_name=task
            )
        elif model_type == 'bert':
            classifier = BertClassifier(
                model_name='bert-base-uncased',
                num_labels=2,
                task_name=task
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'roberta', 'deberta', or 'bert'")
        
        self.model = classifier.get_model()
        self.tokenizer = classifier.get_tokenizer()
        self.label2id = {'literal': 0, task: 1}
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def load_data(self, train_path, val_path, test_path):
        """Load and prepare datasets"""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Filter for binary task
        train_df = train_df[train_df['label'].isin(['literal', self.task])]
        val_df = val_df[val_df['label'].isin(['literal', self.task])]
        test_df = test_df[test_df['label'].isin(['literal', self.task])]
        
        # Convert labels
        for df in [train_df, val_df, test_df]:
            df['label_id'] = df['label'].map(self.label2id)
        
        print(f"\n{self.task.upper()} Detection - Data Loaded:")
        print(f"Train: {len(train_df)}")
        print(f"Val: {len(val_df)}")
        print(f"Test: {len(test_df)}")
        print(f"\nLabel distribution (train):")
        print(train_df['label'].value_counts())
        
        return self.prepare_datasets(train_df, val_df, test_df)
    
    def prepare_datasets(self, train_df, val_df, test_df):
        """Create HuggingFace datasets"""
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df[['text', 'label_id']]),
            'validation': Dataset.from_pandas(val_df[['text', 'label_id']]),
            'test': Dataset.from_pandas(test_df[['text', 'label_id']])
        })
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
        
        tokenized_datasets = dataset_dict.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        tokenized_datasets = tokenized_datasets.rename_column('label_id', 'labels')
        
        return tokenized_datasets
    
    def compute_metrics(self, eval_pred):
        """Compute metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1
        )
        
        macro_f1 = f1_score(labels, predictions, average='macro')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': macro_f1
        }
    
    def calculate_class_weights(self, tokenized_datasets):
        """
        Dynamically calculate class weights from training data
        
        Args:
            tokenized_datasets: Tokenized dataset dict
        
        Returns:
            torch.Tensor with class weights
        """
        # Get training labels
        train_labels = np.array(tokenized_datasets['train']['labels'])
        
        # Calculate class distribution
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        
        print(f"\n{'='*70}")
        print("CLASS DISTRIBUTION & WEIGHTING")
        print(f"{'='*70}")
        print(f"\nClass distribution in training data:")
        for label, count in zip(unique_labels, counts):
            label_name = self.id2label[label]
            percentage = (count / len(train_labels)) * 100
            print(f"  {label_name}: {count} samples ({percentage:.1f}%)")
        
        # Compute balanced class weights using sklearn
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=train_labels
        )
        
        # Convert to torch tensor
        class_weights = torch.tensor(class_weights_array, dtype=torch.float32)
        
        print(f"\nCalculated class weights (balanced):")
        for label, weight in zip(unique_labels, class_weights_array):
            label_name = self.id2label[label]
            print(f"  {label_name}: {weight:.4f}")
        
        print(f"\nThis will make the model pay {class_weights[1]/class_weights[0]:.2f}x more attention to {self.task}")
        print(f"{'='*70}\n")
        
        return class_weights
    
    def train(self, tokenized_datasets, output_dir, 
              num_epochs=5, batch_size=16, learning_rate=2e-5):
        """Train the model with class weighting"""
        
        # Calculate class weights dynamically
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = self.calculate_class_weights(tokenized_datasets).to(device)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy='steps',
            eval_steps=200,
            save_strategy='steps',
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='macro_f1',
            greater_is_better=True,
            report_to='none',
            seed=42
        )
        
        # Use custom WeightedTrainer with class weights
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            class_weights=class_weights  # Pass class weights
        )
        
        print(f"\n{'='*50}")
        print(f"Training {self.task.upper()} Detection Model (WITH CLASS WEIGHTING)")
        print(f"{'='*50}")
        print(f"Device: {device}")
        print(f"{'='*50}\n")
        
        trainer.train()
        
        # Save
        trainer.save_model(f'{output_dir}/final_model')
        self.tokenizer.save_pretrained(f'{output_dir}/final_model')
        
        return trainer
    
    def evaluate(self, trainer, tokenized_datasets, output_dir):
        """Evaluate on test set and save all metrics"""
        print(f"\n{'='*50}")
        print(f"Evaluating {self.task.upper()} Detection")
        print(f"{'='*50}")
        
        predictions = trainer.predict(tokenized_datasets['test'])
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        probs = predictions.predictions  # Get probabilities
        
        # Create results directory
        results_dir = f'{output_dir}/results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Classification report (dict format for saving)
        report_dict = classification_report(
            labels, preds,
            target_names=['Literal', self.task.capitalize()],
            digits=4,
            output_dict=True
        )
        
        # Print classification report
        print("\nTest Set Results:")
        print(classification_report(
            labels, preds,
            target_names=['Literal', self.task.capitalize()],
            digits=4
        ))
        
        # Save classification report as CSV
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f'{results_dir}/classification_report.csv')
        print(f"\n✓ Saved classification report to {results_dir}/classification_report.csv")
        
        # Detailed per-class breakdown
        print(f"\n{'='*50}")
        print("PER-CLASS BREAKDOWN")
        print(f"{'='*50}")
        
        per_class_metrics = []
        for label_id, label_name in self.id2label.items():
            mask = labels == label_id
            if mask.sum() > 0:
                class_preds = preds[mask]
                class_labels = labels[mask]
                correct = (class_preds == class_labels).sum()
                total = len(class_labels)
                accuracy = correct / total
                
                # Calculate precision, recall, f1 for this class
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, labels=[label_id], average='binary', zero_division=0
                )
                
                print(f"\n{label_name.capitalize()}:")
                print(f"  Total samples: {total}")
                print(f"  Correctly predicted: {correct}")
                print(f"  Class accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
                per_class_metrics.append({
                    'class': label_name,
                    'support': int(total),
                    'correct': int(correct),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
        
        # Save per-class metrics
        per_class_df = pd.DataFrame(per_class_metrics)
        per_class_df.to_csv(f'{results_dir}/per_class_metrics.csv', index=False)
        print(f"\n✓ Saved per-class metrics to {results_dir}/per_class_metrics.csv")
        
        # Calculate overall metrics
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        weighted_f1 = f1_score(labels, preds, average='weighted')
        macro_precision = precision_recall_fscore_support(labels, preds, average='macro')[0]
        macro_recall = precision_recall_fscore_support(labels, preds, average='macro')[1]
        
        # Overall metrics summary
        overall_metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'total_samples': len(labels),
            'correct_predictions': (preds == labels).sum()
        }
        
        # Save overall metrics
        overall_df = pd.DataFrame([overall_metrics])
        overall_df.to_csv(f'{results_dir}/overall_metrics.csv', index=False)
        print(f"✓ Saved overall metrics to {results_dir}/overall_metrics.csv")
        
        # Save confusion matrix
        cm = confusion_matrix(labels, preds)
        cm_df = pd.DataFrame(
            cm,
            index=['True_Literal', f'True_{self.task.capitalize()}'],
            columns=['Pred_Literal', f'Pred_{self.task.capitalize()}']
        )
        cm_df.to_csv(f'{results_dir}/confusion_matrix.csv')
        print(f"✓ Saved confusion matrix to {results_dir}/confusion_matrix.csv")
        
        # Save detailed predictions with probabilities
        detailed_predictions = pd.DataFrame({
            'true_label': [self.id2label[l] for l in labels],
            'pred_label': [self.id2label[p] for p in preds],
            'correct': preds == labels,
            'prob_literal': probs[:, 0],
            'prob_figurative': probs[:, 1],
            'confidence': np.max(probs, axis=1)
        })
        detailed_predictions.to_csv(f'{results_dir}/detailed_predictions.csv', index=False)
        print(f"✓ Saved detailed predictions to {results_dir}/detailed_predictions.csv")
        
        # Create text summary report
        summary_text = f"""
{'='*70}
MODEL EVALUATION SUMMARY
{'='*70}

Model: {output_dir}
Task: {self.task.capitalize()} Detection
Test Samples: {len(labels)}

OVERALL METRICS:
  Accuracy:        {accuracy:.4f}
  Macro F1:        {macro_f1:.4f}
  Weighted F1:     {weighted_f1:.4f}
  Macro Precision: {macro_precision:.4f}
  Macro Recall:    {macro_recall:.4f}

CONFUSION MATRIX:
                    Predicted Literal  Predicted {self.task.capitalize()}
True Literal        {cm[0,0]:>16}  {cm[0,1]:>20}
True {self.task.capitalize():<8}    {cm[1,0]:>16}  {cm[1,1]:>20}

PER-CLASS METRICS:
"""
        for metrics in per_class_metrics:
            summary_text += f"\n{metrics['class'].capitalize()}:\n"
            summary_text += f"  Support:   {metrics['support']}\n"
            summary_text += f"  Precision: {metrics['precision']:.4f}\n"
            summary_text += f"  Recall:    {metrics['recall']:.4f}\n"
            summary_text += f"  F1-Score:  {metrics['f1_score']:.4f}\n"
        
        summary_text += f"\n{'='*70}\n"
        summary_text += f"Files saved in: {results_dir}/\n"
        summary_text += f"{'='*70}\n"
        
        # Save summary text
        with open(f'{results_dir}/evaluation_summary.txt', 'w') as f:
            f.write(summary_text)
        print(f"✓ Saved evaluation summary to {results_dir}/evaluation_summary.txt")
        
        print(f"\n{'='*50}")
        print(f"MACRO F1 SCORE: {macro_f1:.4f}")
        print(f"{'='*50}\n")
        
        # Print summary of saved files
        print(f"{'='*70}")
        print("SAVED FILES")
        print(f"{'='*70}")
        print(f"  1. {results_dir}/classification_report.csv")
        print(f"  2. {results_dir}/per_class_metrics.csv")
        print(f"  3. {results_dir}/overall_metrics.csv")
        print(f"  4. {results_dir}/confusion_matrix.csv")
        print(f"  5. {results_dir}/detailed_predictions.csv")
        print(f"  6. {results_dir}/evaluation_summary.txt")
        print(f"{'='*70}\n")
        
        return {
            'predictions': preds,
            'labels': labels,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'report': report_dict,
            'confusion_matrix': cm
        }


def run_experiment_1():
    """
    Experiment 1: Baseline Fine-tuning
    Train separate models for metaphor and irony detection
    """
    print("="*60)
    print("EXPERIMENT 1: BASELINE FINE-TUNING (WITH CLASS WEIGHTING)")
    print("="*60)
    
    # 1. Train metaphor detector on VUA
    print("\n[1/2] Training Metaphor Detector (RoBERTa on VUA)...")
    metaphor_trainer = BaselineTrainer(model_type='roberta', task='metaphor')
    
    # Assuming VUA has been processed to train/val/test splits
    metaphor_datasets = metaphor_trainer.load_data(
        'data/processed/vua_train.csv',
        'data/processed/vua_val.csv',
        'data/processed/vua_test.csv'
    )
    
    metaphor_model = metaphor_trainer.train(
        metaphor_datasets,
        output_dir='models/baseline/roberta_metaphor',
        num_epochs=5,
        batch_size=16
    )
    
    metaphor_results = metaphor_trainer.evaluate(
        metaphor_model,
        metaphor_datasets,
        output_dir='models/baseline/roberta_metaphor'
    )
    
    # 2. Train metaphor detector with DeBERTa
    print("\n[2/2] Training Metaphor Detector (DeBERTa on VUA)...")
    deberta_metaphor_trainer = BaselineTrainer(model_type='deberta', task='metaphor')
    
    metaphor_datasets_deberta = deberta_metaphor_trainer.load_data(
        'data/processed/vua_train.csv',
        'data/processed/vua_val.csv',
        'data/processed/vua_test.csv'
    )
    
    deberta_metaphor_model = deberta_metaphor_trainer.train(
        metaphor_datasets_deberta,
        output_dir='models/baseline/deberta_metaphor',
        num_epochs=5,
        batch_size=16
    )
    
    deberta_metaphor_results = deberta_metaphor_trainer.evaluate(
        deberta_metaphor_model,
        metaphor_datasets_deberta,
        output_dir='models/baseline/deberta_metaphor'
    )
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT 1 COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nRoBERTa Metaphor Macro F1: {metaphor_results['macro_f1']:.4f}")
    print(f"DeBERTa Metaphor Macro F1: {deberta_metaphor_results['macro_f1']:.4f}")
    print("\n" + "="*60)

if __name__ == "__main__":
    run_experiment_1()