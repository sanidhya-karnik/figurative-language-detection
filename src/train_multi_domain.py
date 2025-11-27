#!/usr/bin/env python3
"""
Train models on multiple domains: Reddit, IMDb, and News

This script allows you to:
1. Fine-tune VUA model on each domain separately
2. Compare performance across domains
3. Evaluate cross-domain generalization
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.train_baseline import BaselineTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split


def prepare_domain_data(domain_file, domain_name):
    """
    Prepare domain-specific data
    
    Args:
        domain_file: Path to annotated data (CSV or XLSX)
        domain_name: Name of domain (reddit, imdb, news)
    """
    print(f"\n{'='*70}")
    print(f"PREPARING {domain_name.upper()} DATA")
    print(f"{'='*70}")
    
    # Load data
    if domain_file.endswith('.xlsx'):
        df = pd.read_excel(domain_file)
    else:
        df = pd.read_csv(domain_file)
    
    print(f"Loaded {len(df)} samples")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Extract text column based on domain
    if domain_name == 'imdb':
        # Combine review_title and review_body
        df['text'] = df['review_title'].fillna('') + ' ' + df['review_body'].fillna('')
        df['text'] = df['text'].str.strip()
    elif domain_name == 'news':
        # Use 'text' column (or 'title' + 'text' if you prefer)
        if 'text' in df.columns:
            df['text'] = df['text'].fillna('')
        else:
            df['text'] = df['title'].fillna('') + ' ' + df.get('text', '').fillna('')
    elif domain_name == 'reddit':
        # Text column already exists
        if 'text' not in df.columns:
            print("❌ 'text' column not found!")
            return None
    
    # Check for label column
    if 'label' not in df.columns:
        print("❌ 'label' column not found!")
        return None
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10].copy()
    
    print(f"\nAfter filtering: {len(df)} samples")
    print(f"\nOriginal label distribution:")
    print(df['label'].value_counts())
    
    # Convert to binary labels (literal vs figurative)
    df['label'] = df['label'].apply(
        lambda x: 'literal' if x == 'literal' else 'figurative'
    )
    
    print(f"\nBinary label distribution:")
    print(df['label'].value_counts())
    
    # Check if we have both classes
    if len(df['label'].unique()) < 2:
        print(f"\n❌ Only one class found! Cannot train.")
        return None
    
    # Split: 60% train, 20% val, 20% test
    train, temp = train_test_split(
        df,
        test_size=0.4,
        stratify=df['label'],
        random_state=42
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp['label'],
        random_state=42
    )
    
    print(f"\nSplits:")
    print(f"  Train: {len(train)} (literal: {(train['label']=='literal').sum()}, figurative: {(train['label']=='figurative').sum()})")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
    
    # Save
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train[['text', 'label']].to_csv(output_dir / f'{domain_name}_train.csv', index=False)
    val[['text', 'label']].to_csv(output_dir / f'{domain_name}_val.csv', index=False)
    test[['text', 'label']].to_csv(output_dir / f'{domain_name}_test.csv', index=False)
    
    print(f"\n✓ Saved {domain_name} splits to data/processed/")
    
    return {
        'train': len(train),
        'val': len(val),
        'test': len(test)
    }


def train_on_domain(model_type, domain_name, vua_model_path=None):
    """
    Fine-tune on a specific domain
    
    Args:
        model_type: 'roberta', 'deberta', or 'bert'
        domain_name: 'reddit', 'imdb', or 'news'
        vua_model_path: Path to VUA-trained model (if None, trains from scratch)
    """
    print(f"\n{'='*70}")
    print(f"FINE-TUNING {model_type.upper()} ON {domain_name.upper()}")
    print(f"{'='*70}")
    
    # Initialize trainer
    trainer = BaselineTrainer(model_type=model_type, task='metaphor')
    
    # Load VUA-trained model if provided
    if vua_model_path and Path(vua_model_path).exists():
        print(f"\n✓ Loading VUA-trained model from {vua_model_path}")
        trainer.model = AutoModelForSequenceClassification.from_pretrained(vua_model_path)
        trainer.tokenizer = AutoTokenizer.from_pretrained(vua_model_path)
    else:
        print(f"\n⚠️  Training from scratch (no VUA model found)")
    
    # Load domain data
    datasets = trainer.load_data(
        f'data/processed/{domain_name}_train.csv',
        f'data/processed/{domain_name}_val.csv',
        f'data/processed/{domain_name}_test.csv'
    )
    
    # Train
    output_dir = f'models/{domain_name}_finetuned_{model_type}'
    
    trained_model = trainer.train(
        datasets,
        output_dir=output_dir,
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-5  # Lower LR for fine-tuning
    )
    
    # Evaluate
    results = trainer.evaluate(trained_model, datasets, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ {domain_name.upper()} Model Complete!")
    print(f"{'='*70}")
    print(f"Test Macro F1: {results['macro_f1']:.4f}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return results


def train_all_domains(model_type='roberta'):
    """Train on all three domains"""
    
    print("="*70)
    print("MULTI-DOMAIN TRAINING")
    print("="*70)
    print(f"\nModel: {model_type.upper()}")
    print("Domains: Reddit, IMDb, News")
    print("\nThis will:")
    print("  1. Prepare data splits for each domain")
    print("  2. Fine-tune VUA model on each domain separately")
    print("  3. Evaluate on each domain's test set")
    print("  4. Compare results across domains")
    print("="*70)
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # VUA model path
    vua_model_path = f'models/vua_baseline_{model_type}/final_model'
    
    if not Path(vua_model_path).exists():
        print(f"\n⚠️  VUA model not found at {vua_model_path}")
        print("Training from scratch on each domain...")
        vua_model_path = None
    
    # Prepare all domains
    domains = {
        'reddit': 'data/processed/reddit_annotations.csv',
        'imdb': 'data/processed/imdb_reviews_annotated.csv',
        'news': 'data/processed/news_articles_annotated.csv'
    }
    
    # Check which files exist
    available_domains = {}
    for domain, filepath in domains.items():
        if Path(filepath).exists():
            available_domains[domain] = filepath
        else:
            print(f"\n⚠️  {domain} data not found at {filepath}")
    
    if not available_domains:
        print("\n❌ No domain data found!")
        return
    
    # Prepare data for each domain
    print(f"\n{'>'*70}")
    print("STEP 1: PREPARING DATA")
    print(f"{'>'*70}")
    
    for domain, filepath in available_domains.items():
        prepare_domain_data(filepath, domain)
    
    # Train on each domain
    print(f"\n{'>'*70}")
    print("STEP 2: TRAINING ON EACH DOMAIN")
    print(f"{'>'*70}")
    
    results = {}
    
    for domain in available_domains.keys():
        results[domain] = train_on_domain(model_type, domain, vua_model_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("MULTI-DOMAIN TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}")
    
    for domain, result in results.items():
        print(f"\n{domain.capitalize()}:")
        print(f"  Macro F1: {result['macro_f1']:.4f}")
        print(f"  Accuracy: {result.get('accuracy', 'N/A')}")
    
    # Find best
    best_domain = max(results.items(), key=lambda x: x[1]['macro_f1'])
    print(f"\n🏆 Best performing domain: {best_domain[0].upper()} (F1: {best_domain[1]['macro_f1']:.4f})")
    
    # Save comparison
    comparison_df = pd.DataFrame([
        {
            'Domain': domain,
            'Macro_F1': result['macro_f1'],
            'Model': f'{domain}_finetuned_{model_type}'
        }
        for domain, result in results.items()
    ])
    
    comparison_df.to_csv('results/multi_domain_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to results/multi_domain_comparison.csv")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-domain training')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['roberta', 'deberta', 'bert'],
        default='roberta',
        help='Model type to use'
    )
    parser.add_argument(
        '--domain',
        type=str,
        choices=['reddit', 'imdb', 'news', 'all'],
        default='all',
        help='Which domain to train on'
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare data splits, do not train'
    )
    
    args = parser.parse_args()
    
    if args.prepare_only:
        # Just prepare data
        domains = {
            'reddit': 'data/processed/reddit_annotations.csv',
            'imdb': 'data/processed/imdb_reviews_annotated.csv',
            'news': 'data/processed/news_articles_annotated.csv'
        }
        
        for domain, filepath in domains.items():
            if Path(filepath).exists():
                prepare_domain_data(filepath, domain)
    
    elif args.domain == 'all':
        # Train on all domains
        train_all_domains(args.model)
    
    else:
        # Train on specific domain
        domain_file = f'data/processed/{args.domain}_annotations.csv'
        
        if not Path(domain_file).exists():
            # Try without _annotations
            domain_file = f'data/processed/{args.domain}_annotated.csv'
        
        if Path(domain_file).exists():
            # Prepare data
            prepare_domain_data(domain_file, args.domain)
            
            # Train
            vua_model_path = f'models/vua_baseline_{args.model}/final_model'
            train_on_domain(args.model, args.domain, vua_model_path)
        else:
            print(f"❌ Data not found for {args.domain}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)