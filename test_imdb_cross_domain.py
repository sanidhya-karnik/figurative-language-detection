#!/usr/bin/env python3
"""
Test IMDB fine-tuned models (binary: literal vs figurative) on full Reddit and News annotations.

The IMDB models were trained on binary classification (literal vs figurative).
This script tests them on the full annotated data which has 4 labels: literal, metaphor, irony, both.

We map the labels:
- literal → literal (0)
- metaphor, irony, both → figurative (1)

Usage:
    # Test specific model
    python test_imdb_cross_domain.py --model_name roberta
    
    # Test all IMDB models
    python test_imdb_cross_domain.py --test_all
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class FigurativeLanguageDataset(Dataset):
    """Dataset for figurative language detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class CrossDomainTester:
    """Test binary model (literal vs figurative) across different domains"""
    
    def __init__(self, model_path, model_name, device=None):
        """
        Initialize tester with binary classification model
        
        Args:
            model_path: Path to saved model checkpoint
            model_name: Name of the model (e.g., 'roberta', 'bert', 'deberta')
            device: torch device (auto-detect if None)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Binary label mappings for the model
        self.binary_label2id = {'literal': 0, 'figurative': 1}
        self.binary_id2label = {v: k for k, v in self.binary_label2id.items()}
        
        # Load model and tokenizer
        self._load_model()
        
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Model type: Binary (literal vs figurative)")
        print(f"✓ Model path: {model_path}")
        print(f"✓ Using device: {self.device}")
    
    def _load_model(self):
        """Load model and tokenizer from checkpoint"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Check if model is binary
            if self.model.config.num_labels != 2:
                print(f"⚠ Warning: Model has {self.model.config.num_labels} labels, expected 2 for binary classification")
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")
    
    def map_to_binary(self, label):
        """
        Map 4-class labels to binary
        
        Args:
            label: One of 'literal', 'metaphor', 'irony', 'both'
        
        Returns:
            Binary label: 'literal' or 'figurative'
        """
        if label == 'literal':
            return 'literal'
        else:
            return 'figurative'
    
    def load_data(self, data_path, domain_name):
        """
        Load and prepare test data
        
        Args:
            data_path: Path to CSV file
            domain_name: Name of domain (for logging)
        
        Returns:
            DataFrame with loaded and processed data
        """
        print(f"\n{'='*60}")
        print(f"Loading {domain_name} annotations...")
        print(f"{'='*60}")
        
        df = pd.read_csv(data_path)
        
        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Data must have 'text' and 'label' columns. Found: {df.columns.tolist()}")
        
        print(f"Total samples: {len(df)}")
        print(f"\nOriginal label distribution (4-class):")
        print(df['label'].value_counts())
        
        # Store original labels
        df['original_label'] = df['label']
        
        # Map to binary labels
        df['binary_label'] = df['original_label'].apply(self.map_to_binary)
        df['binary_label_id'] = df['binary_label'].map(self.binary_label2id)
        
        print(f"\nMapped to binary labels:")
        print(df['binary_label'].value_counts())
        print(f"\nMapping:")
        print(f"  literal → literal")
        print(f"  metaphor, irony, both → figurative")
        
        # Remove any rows with invalid labels
        invalid = df['binary_label_id'].isna()
        if invalid.any():
            print(f"\n⚠ Warning: Removing {invalid.sum()} rows with invalid labels")
            df = df[~invalid].copy()
        
        return df
    
    def predict(self, df, batch_size=16):
        """
        Generate predictions for dataset
        
        Args:
            df: DataFrame with text and labels
            batch_size: Batch size for inference
        
        Returns:
            predictions: List of predicted binary labels
            probabilities: Array of prediction probabilities (2 classes)
        """
        dataset = FigurativeLanguageDataset(
            texts=df['text'].tolist(),
            labels=df['binary_label_id'].tolist(),
            tokenizer=self.tokenizer
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Convert to binary label names
        pred_labels = [self.binary_id2label[p] for p in all_predictions]
        
        return pred_labels, np.array(all_probabilities)
    
    def evaluate_binary(self, true_binary_labels, pred_binary_labels, domain_name):
        """
        Evaluate binary predictions (literal vs figurative)
        
        Args:
            true_binary_labels: True binary label names
            pred_binary_labels: Predicted binary label names
            domain_name: Name of domain
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*60}")
        print(f"{domain_name.upper()} - BINARY PERFORMANCE")
        print(f"Model: {self.model_name.upper()} (IMDB-trained)")
        print(f"{'='*60}")
        
        # Overall metrics
        accuracy = accuracy_score(true_binary_labels, pred_binary_labels)
        f1_macro = f1_score(true_binary_labels, pred_binary_labels, average='macro')
        f1_weighted = f1_score(true_binary_labels, pred_binary_labels, average='weighted')
        
        print(f"\n📊 Binary Classification Metrics (Literal vs Figurative):")
        print(f"  Accuracy:      {accuracy:.4f}")
        print(f"  F1 (Macro):    {f1_macro:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")
        
        # Per-class metrics
        precision, recall, f1_class, support = precision_recall_fscore_support(
            true_binary_labels, pred_binary_labels, labels=['literal', 'figurative'], zero_division=0
        )
        
        print(f"\n📈 Per-Class Metrics:")
        for i, label in enumerate(['literal', 'figurative']):
            print(f"  {label.capitalize():12s}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1_class[i]:.4f}, N={int(support[i])}")
        
        # Detailed classification report
        print(f"\n{'-'*60}")
        print("Classification Report:")
        print(f"{'-'*60}")
        print(classification_report(true_binary_labels, pred_binary_labels, digits=4))
        
        # Store metrics
        metrics = {
            'model': self.model_name,
            'task': 'binary',
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'per_class': {
                'literal': {
                    'precision': float(precision[0]), 
                    'recall': float(recall[0]), 
                    'f1': float(f1_class[0]), 
                    'support': int(support[0])
                },
                'figurative': {
                    'precision': float(precision[1]), 
                    'recall': float(recall[1]), 
                    'f1': float(f1_class[1]), 
                    'support': int(support[1])
                }
            }
        }
        
        return metrics
    
    def evaluate_by_original_label(self, df, pred_binary_labels, domain_name):
        """
        Break down performance by original 4-class labels
        
        Args:
            df: DataFrame with original_label and binary predictions
            pred_binary_labels: Predicted binary labels
            domain_name: Name of domain
        """
        print(f"\n{'='*60}")
        print(f"{domain_name.upper()} - BREAKDOWN BY ORIGINAL LABEL")
        print(f"{'='*60}")
        
        df_analysis = df.copy()
        df_analysis['predicted_binary'] = pred_binary_labels
        df_analysis['correct'] = df_analysis['binary_label'] == df_analysis['predicted_binary']
        
        print(f"\nPerformance by original label type:")
        print(f"{'Label':<12s} {'Total':<8s} {'Correct':<8s} {'Accuracy':<10s} {'Notes'}")
        print("-" * 70)
        
        breakdown = {}
        for original_label in ['literal', 'metaphor', 'irony', 'both']:
            subset = df_analysis[df_analysis['original_label'] == original_label]
            if len(subset) > 0:
                correct = subset['correct'].sum()
                total = len(subset)
                acc = correct / total if total > 0 else 0
                
                # Get binary mapping
                binary_class = 'literal' if original_label == 'literal' else 'figurative'
                
                print(f"{original_label:<12s} {total:<8d} {correct:<8d} {acc:<10.4f} ({binary_class})")
                
                breakdown[original_label] = {
                    'total': int(total),
                    'correct': int(correct),
                    'accuracy': float(acc),
                    'maps_to': binary_class
                }
        
        print(f"\n{'Total':<12s} {len(df_analysis):<8d} {df_analysis['correct'].sum():<8d} {df_analysis['correct'].mean():<10.4f}")
        
        return breakdown
    
    def plot_confusion_matrix(self, true_labels, pred_labels, domain_name, save_path=None):
        """Plot binary confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels, labels=['literal', 'figurative'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Literal', 'Figurative'],
                    yticklabels=['Literal', 'Figurative'],
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('True', fontsize=12, fontweight='bold')
        plt.title(f'Binary Confusion Matrix - {domain_name}\n{self.model_name.upper()} (IMDB-trained)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def plot_original_label_breakdown(self, breakdown, domain_name, save_path=None):
        """Plot accuracy breakdown by original 4-class labels"""
        labels = list(breakdown.keys())
        accuracies = [breakdown[label]['accuracy'] for label in labels]
        totals = [breakdown[label]['total'] for label in labels]
        
        # Color code by binary mapping
        colors = ['#3498db' if label == 'literal' else '#e74c3c' for label in labels]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, acc, total in zip(bars, accuracies, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}\n(n={total})',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Original Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Binary Model Performance by Original Label Type\n{domain_name} - {self.model_name.upper()}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Maps to Literal'),
            Patch(facecolor='#e74c3c', label='Maps to Figurative')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved breakdown plot to {save_path}")
        
        plt.close()
    
    def analyze_errors(self, df, pred_labels, domain_name, n_examples=3):
        """Analyze prediction errors"""
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS - {domain_name.upper()}")
        print(f"{'='*60}")
        
        df_analysis = df.copy()
        df_analysis['predicted_binary'] = pred_labels
        df_analysis['correct'] = df_analysis['binary_label'] == df_analysis['predicted_binary']
        
        errors = df_analysis[~df_analysis['correct']]
        
        print(f"\nTotal errors: {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.2f}%)")
        
        # Errors by binary label
        print(f"\nBinary errors:")
        for binary_label in ['literal', 'figurative']:
            label_total = len(df_analysis[df_analysis['binary_label'] == binary_label])
            label_errors = len(errors[errors['binary_label'] == binary_label])
            if label_total > 0:
                print(f"  {binary_label.capitalize():12s}: {label_errors:3d} / {label_total:3d} ({label_errors/label_total*100:5.2f}%)")
        
        # Errors by original label
        print(f"\nErrors by original label type:")
        for orig_label in ['literal', 'metaphor', 'irony', 'both']:
            label_errors = errors[errors['original_label'] == orig_label]
            if len(label_errors) > 0:
                label_total = len(df_analysis[df_analysis['original_label'] == orig_label])
                print(f"  {orig_label.capitalize():12s}: {len(label_errors):3d} / {label_total:3d} ({len(label_errors)/label_total*100:5.2f}%)")
        
        # Show error examples
        print(f"\n{'-'*60}")
        print("Sample Errors:")
        print(f"{'-'*60}")
        
        for binary_true in ['literal', 'figurative']:
            binary_errors = errors[errors['binary_label'] == binary_true]
            if len(binary_errors) > 0:
                print(f"\n{binary_true.upper()} misclassified as:")
                
                # Group by original label
                for orig_label in ['literal', 'metaphor', 'irony', 'both']:
                    orig_errors = binary_errors[binary_errors['original_label'] == orig_label]
                    if len(orig_errors) > 0:
                        print(f"\n  From original '{orig_label}' ({len(orig_errors)} cases):")
                        examples = orig_errors.head(n_examples)
                        for i, (_, row) in enumerate(examples.iterrows(), 1):
                            text = row['text'][:100] + '...' if len(row['text']) > 100 else row['text']
                            print(f"    {i}. \"{text}\"")


def plot_domain_comparison(results, save_path=None):
    """Plot comparison across domains for binary classification"""
    domains = list(results.keys())
    model_name = results[domains[0]]['model']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_names = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)']
    metrics_keys = ['accuracy', 'f1_macro', 'f1_weighted']
    colors = ['#3498db', '#2ecc71']
    
    for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
        ax = axes[idx]
        scores = [results[domain][metric_key] for domain in domains]
        
        bars = ax.bar(domains, scores, color=colors[:len(domains)], alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.suptitle(f'{model_name.upper()} Binary Classification: Reddit vs News', 
                 y=1.02, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot to {save_path}")
    
    plt.close()


def plot_per_class_comparison(results, save_path=None):
    """Plot per-class performance comparison for binary classification"""
    domains = list(results.keys())
    classes = ['literal', 'figurative']
    model_name = results[domains[0]]['model']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_keys = ['precision', 'recall', 'f1']
    
    colors = ['#3498db', '#2ecc71']
    
    for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
        ax = axes[idx]
        x = np.arange(len(classes))
        width = 0.35
        
        for i, domain in enumerate(domains):
            values = [results[domain]['per_class'][cls][metric_key] for cls in classes]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=domain, 
                          color=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in classes], fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.suptitle(f'{model_name.upper()} Per-Class Performance: Reddit vs News', 
                 y=1.02, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved per-class plot to {save_path}")
    
    plt.close()


def test_single_model(model_name, reddit_data, news_data, output_dir, batch_size=16):
    """Test a single IMDB binary model on Reddit and News annotations"""
    
    # Model path
    model_path = f"./models/imdb_finetuned_{model_name}/final_model"
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return None
    
    print("\n" + "="*70)
    print(f"TESTING: IMDB-{model_name.upper()} (Binary) → Full Annotations")
    print("="*70)
    
    # Create output directory for this model
    model_output_dir = os.path.join(output_dir, f"imdb_{model_name}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize tester
    tester = CrossDomainTester(model_path, model_name)
    
    # Test on both domains
    results = {}
    all_breakdowns = {}
    
    for domain_name, data_path in [('Reddit', reddit_data), ('News', news_data)]:
        # Load data
        df = tester.load_data(data_path, domain_name)
        
        # Generate predictions
        pred_binary_labels, pred_probs = tester.predict(df, batch_size=batch_size)
        
        # Evaluate binary performance
        metrics = tester.evaluate_binary(
            df['binary_label'].tolist(), 
            pred_binary_labels, 
            domain_name
        )
        results[domain_name] = metrics
        
        # Breakdown by original labels
        breakdown = tester.evaluate_by_original_label(df, pred_binary_labels, domain_name)
        all_breakdowns[domain_name] = breakdown
        
        # Plot confusion matrix (binary)
        cm_path = os.path.join(model_output_dir, f'binary_cm_{domain_name.lower()}.png')
        tester.plot_confusion_matrix(
            df['binary_label'].tolist(), 
            pred_binary_labels, 
            domain_name, 
            save_path=cm_path
        )
        
        # Plot original label breakdown
        breakdown_path = os.path.join(model_output_dir, f'breakdown_{domain_name.lower()}.png')
        tester.plot_original_label_breakdown(breakdown, domain_name, save_path=breakdown_path)
        
        # Analyze errors
        tester.analyze_errors(df, pred_binary_labels, domain_name)
        
        # Save predictions
        df['predicted_binary'] = pred_binary_labels
        df['prob_literal'] = pred_probs[:, 0]
        df['prob_figurative'] = pred_probs[:, 1]
        df['confidence'] = pred_probs.max(axis=1)
        df['correct'] = df['binary_label'] == df['predicted_binary']
        
        pred_path = os.path.join(model_output_dir, f'{domain_name.lower()}_predictions.csv')
        df.to_csv(pred_path, index=False)
        print(f"  ✓ Saved predictions to {pred_path}")
    
    # Compare domains
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN COMPARISON (Binary Classification)")
    print(f"{'='*60}")
    print(f"\nModel: {model_name.upper()}")
    print(f"{'':15s} Reddit    News      Difference")
    print(f"{'-'*50}")
    print(f"Accuracy:       {results['Reddit']['accuracy']:.4f}   {results['News']['accuracy']:.4f}    {abs(results['Reddit']['accuracy']-results['News']['accuracy']):.4f}")
    print(f"F1 (Macro):     {results['Reddit']['f1_macro']:.4f}   {results['News']['f1_macro']:.4f}    {abs(results['Reddit']['f1_macro']-results['News']['f1_macro']):.4f}")
    print(f"F1 (Weighted):  {results['Reddit']['f1_weighted']:.4f}   {results['News']['f1_weighted']:.4f}    {abs(results['Reddit']['f1_weighted']-results['News']['f1_weighted']):.4f}")
    
    # Determine better domain
    if results['Reddit']['accuracy'] > results['News']['accuracy']:
        better_domain = 'Reddit'
        improvement = ((results['Reddit']['accuracy'] - results['News']['accuracy']) / results['News']['accuracy'] * 100)
    else:
        better_domain = 'News'
        improvement = ((results['News']['accuracy'] - results['Reddit']['accuracy']) / results['Reddit']['accuracy'] * 100)
    
    print(f"\n→ Model performs {improvement:.2f}% better on {better_domain}")
    
    # Plot comparisons
    print(f"\nGenerating visualizations...")
    comparison_path = os.path.join(model_output_dir, 'domain_comparison.png')
    plot_domain_comparison(results, save_path=comparison_path)
    
    per_class_path = os.path.join(model_output_dir, 'per_class_comparison.png')
    plot_per_class_comparison(results, save_path=per_class_path)
    
    # Save comprehensive summary JSON
    summary = {
        'model': model_name,
        'model_path': model_path,
        'task': 'binary_classification',
        'label_mapping': {
            'literal': 'literal',
            'metaphor': 'figurative',
            'irony': 'figurative',
            'both': 'figurative'
        },
        'reddit': {
            'binary_metrics': results['Reddit'],
            'breakdown_by_original_label': all_breakdowns['Reddit']
        },
        'news': {
            'binary_metrics': results['News'],
            'breakdown_by_original_label': all_breakdowns['News']
        },
        'comparison': {
            'accuracy_diff': abs(results['Reddit']['accuracy'] - results['News']['accuracy']),
            'f1_macro_diff': abs(results['Reddit']['f1_macro'] - results['News']['f1_macro']),
            'better_domain': better_domain,
            'improvement_pct': improvement
        }
    }
    
    json_path = os.path.join(model_output_dir, 'results_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved comprehensive summary to {json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test IMDB binary models on full Reddit and News annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test RoBERTa model on full annotations
  python test_imdb_cross_domain.py --model_name roberta
  
  # Test all models
  python test_imdb_cross_domain.py --test_all
  
  # Custom paths
  python test_imdb_cross_domain.py --model_name bert \\
      --reddit_data data/processed/reddit_annotations.csv \\
      --news_data data/processed/news_articles_annotated.csv
        """
    )
    
    parser.add_argument('--model_name', type=str, choices=['roberta', 'bert', 'deberta'],
                        help='Which IMDB model to test')
    parser.add_argument('--test_all', action='store_true',
                        help='Test all IMDB models (roberta, bert, deberta)')
    parser.add_argument('--reddit_data', type=str, 
                        default='./data/processed/reddit_annotations.csv',
                        help='Path to Reddit annotations (text, label, notes)')
    parser.add_argument('--news_data', type=str, 
                        default='./data/processed/news_articles_annotated.csv',
                        help='Path to News annotations (source, url, title, text, publish_date, label, notes)')
    parser.add_argument('--output_dir', type=str, default='./results/cross_domain',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Validation
    if not args.model_name and not args.test_all:
        parser.error("Must specify either --model_name or --test_all")
    
    if not os.path.exists(args.reddit_data):
        parser.error(f"Reddit data not found: {args.reddit_data}")
    
    if not os.path.exists(args.news_data):
        parser.error(f"News data not found: {args.news_data}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print(" "*10 + "CROSS-DOMAIN BINARY MODEL TESTING")
    print(" "*5 + "IMDB-trained (literal vs figurative) → Full Annotations")
    print("="*70)
    print(f"\n📂 Data:")
    print(f"  Reddit: {args.reddit_data}")
    print(f"  News:   {args.news_data}")
    print(f"\n📊 Output: {args.output_dir}")
    print(f"\n🏷️  Label Mapping:")
    print(f"  literal → literal")
    print(f"  metaphor, irony, both → figurative")
    
    # Determine which models to test
    if args.test_all:
        models_to_test = ['roberta', 'bert', 'deberta']
    else:
        models_to_test = [args.model_name]
    
    print(f"\n🔬 Testing models: {', '.join([m.upper() for m in models_to_test])}")
    
    # Test each model
    all_results = {}
    for model_name in models_to_test:
        results = test_single_model(
            model_name=model_name,
            reddit_data=args.reddit_data,
            news_data=args.news_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        if results:
            all_results[model_name] = results
    
    # Final summary
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY: ALL MODELS (Binary Classification)")
        print("="*70)
        print(f"\n{'Model':<15s} {'Reddit Acc':<12s} {'News Acc':<12s} {'Reddit F1':<12s} {'News F1':<12s}")
        print("-"*70)
        for model_name, results in all_results.items():
            reddit_acc = results['Reddit']['accuracy']
            news_acc = results['News']['accuracy']
            reddit_f1 = results['Reddit']['f1_macro']
            news_f1 = results['News']['f1_macro']
            print(f"{model_name.upper():<15s} {reddit_acc:<12.4f} {news_acc:<12.4f} {reddit_f1:<12.4f} {news_f1:<12.4f}")
    
    print("\n" + "="*70)
    print(" "*20 + "TESTING COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()