#!/usr/bin/env python3
"""
Quick runner for Reddit figurative language detection pipeline

Usage:
    python run_reddit.py                    # Interactive mode
    python run_reddit.py --step 3           # Run specific step
    python run_reddit.py --train            # Train all models (RoBERTa, DeBERTa, BERT)
    python run_reddit.py --train-roberta    # Train only RoBERTa
    python run_reddit.py --train-deberta    # Train only DeBERTa
    python run_reddit.py --train-bert       # Train only BERT
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def step_1_scrape():
    """Scrape Reddit data"""
    print("\n🔍 STEP 1: Scraping Reddit data...")
    from data_collection.scrape_reddit import main
    main()

def step_2_prepare_vua():
    """Process VUA corpus"""
    print("\n📚 STEP 2: Processing VUA corpus...")
    from preprocessing.prepare_vua import main
    main()

def step_3_train_vua(model_type='roberta'):
    """
    Train on VUA with specified model
    
    Args:
        model_type: 'roberta', 'deberta', or 'bert'
    """
    print(f"\n🎓 STEP 3: Training on VUA corpus with {model_type.upper()}...")
    print(f"This will take 1-2 hours for {model_type.upper()}...")
    
    from training.train_baseline import BaselineTrainer
    
    # Set output directory based on model type
    output_dir = f'models/vua_baseline_{model_type}'
    
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} on VUA Metaphor Corpus")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    trainer = BaselineTrainer(model_type=model_type, task='metaphor')
    datasets = trainer.load_data(
        'data/processed/vua_train.csv',
        'data/processed/vua_val.csv',
        'data/processed/vua_test.csv'
    )
    
    model = trainer.train(
        datasets,
        output_dir=output_dir,
        num_epochs=5,
        batch_size=16
    )
    
    results = trainer.evaluate(model, datasets, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ {model_type.upper()} model trained!")
    print(f"{'='*70}")
    print(f"Test Macro F1: {results['macro_f1']:.4f}")
    print(f"Model saved to: {output_dir}/final_model")
    print(f"{'='*70}\n")
    
    return results

def step_3_train_all_models():
    """Train all three models: RoBERTa, DeBERTa, and BERT"""
    print("\n" + "="*70)
    print("TRAINING ALL MODELS ON VUA")
    print("="*70)
    print("\nThis will train 3 models sequentially:")
    print("  1. RoBERTa-base")
    print("  2. DeBERTa-v3-base")
    print("  3. BERT-base")
    print("\nTotal estimated time: 3-6 hours")
    print("="*70)
    
    response = input("\nProceed with training all models? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    results = {}
    
    # Train RoBERTa
    print("\n" + ">"*70)
    print("TRAINING MODEL 1/3: RoBERTa")
    print(">"*70)
    results['roberta'] = step_3_train_vua('roberta')
    
    # Train DeBERTa
    print("\n" + ">"*70)
    print("TRAINING MODEL 2/3: DeBERTa")
    print(">"*70)
    results['deberta'] = step_3_train_vua('deberta')
    
    # Train BERT
    print("\n" + ">"*70)
    print("TRAINING MODEL 3/3: BERT")
    print(">"*70)
    results['bert'] = step_3_train_vua('bert')
    
    # Summary
    print("\n" + "="*70)
    print("ALL MODELS TRAINED - SUMMARY")
    print("="*70)
    print(f"\nRoBERTa Macro F1:  {results['roberta']['macro_f1']:.4f}")
    print(f"DeBERTa Macro F1:  {results['deberta']['macro_f1']:.4f}")
    print(f"BERT Macro F1:     {results['bert']['macro_f1']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['macro_f1'])
    print(f"\n🏆 Best model: {best_model[0].upper()} (F1: {best_model[1]['macro_f1']:.4f})")
    print("="*70)
    
    return results

def step_4_finetune_reddit(model_type='roberta'):
    """
    Fine-tune on Reddit
    
    Args:
        model_type: 'roberta', 'deberta', or 'bert'
    """
    print(f"\n🎯 STEP 4: Fine-tuning {model_type.upper()} on Reddit data...")
    
    # Check if annotations exist
    if not Path('data/processed/reddit_train.csv').exists():
        print("\n❌ Reddit annotations not found!")
        print("Please:")
        print("1. Annotate data in notebooks/annotation_guide.ipynb")
        print("2. Save to data/annotations/reddit_annotated.csv")
        print("3. Run: python -m src.preprocessing.prepare_custom")
        return
    
    from training.train_baseline import BaselineTrainer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load VUA model
    vua_model_path = f'models/vua_baseline_{model_type}/final_model'
    if not Path(vua_model_path).exists():
        print(f"\n❌ {model_type.upper()} VUA model not found at {vua_model_path}")
        print(f"Please run: python run_reddit.py --train-{model_type}")
        return
    
    trainer = BaselineTrainer(model_type=model_type, task='metaphor')
    trainer.model = AutoModelForSequenceClassification.from_pretrained(vua_model_path)
    trainer.tokenizer = AutoTokenizer.from_pretrained(vua_model_path)
    
    # Load Reddit data
    datasets = trainer.load_data(
        'data/processed/reddit_train.csv',
        'data/processed/reddit_val.csv',
        'data/processed/reddit_test.csv'
    )
    
    # Fine-tune
    output_dir = f'models/reddit_finetuned_{model_type}'
    model = trainer.train(
        datasets,
        output_dir=output_dir,
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-5  # Lower LR for fine-tuning
    )
    
    results = trainer.evaluate(model, datasets, output_dir)
    print(f"\n✅ {model_type.upper()} model fine-tuned! Test F1: {results['macro_f1']:.4f}")
    
    return results

def step_5_evaluate(model_type='roberta'):
    """
    Evaluate model
    
    Args:
        model_type: 'roberta', 'deberta', or 'bert'
    """
    print(f"\n📊 STEP 5: Evaluating {model_type.upper()} model...")
    
    import pandas as pd
    import torch
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load test data
    test_df = pd.read_csv('data/processed/reddit_test.csv')
    
    # Load model
    model_path = f'models/reddit_finetuned_{model_type}/final_model'
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at {model_path}")
        return
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get predictions
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    true_labels = []
    
    print(f"Generating predictions for {len(test_df)} samples...")
    
    for idx, row in test_df.iterrows():
        inputs = tokenizer(
            row['text'],
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)
        
        label_map = {'literal': 0, 'metaphor': 1, 'irony': 1, 'both': 1}
        true_labels.append(label_map.get(row['label'], 0))
    
    # Evaluate
    from sklearn.metrics import classification_report, f1_score
    
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS - {model_type.upper()}")
    print("="*70)
    print(classification_report(
        true_labels, 
        predictions,
        target_names=['Literal', 'Figurative']
    ))
    
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    print(f"\nMacro F1: {macro_f1:.4f}")
    
    print("\n✅ Evaluation complete!")

def interactive_menu():
    """Interactive menu"""
    print("="*70)
    print(" "*15 + "REDDIT FIGURATIVE LANGUAGE DETECTION")
    print("="*70)
    
    print("\nPipeline Steps:")
    print("  1. Scrape Reddit data")
    print("  2. Process VUA corpus")
    print("  3. Train on VUA (select model)")
    print("     3a. Train RoBERTa")
    print("     3b. Train DeBERTa")
    print("     3c. Train BERT")
    print("     3d. Train ALL models")
    print("  4. Fine-tune on Reddit")
    print("  5. Evaluate model")
    print("  Q. Quit")
    
    while True:
        choice = input("\nSelect option: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice == '1':
            step_1_scrape()
        elif choice == '2':
            step_2_prepare_vua()
        elif choice == '3':
            print("\nSelect model:")
            print("  a. RoBERTa")
            print("  b. DeBERTa")
            print("  c. BERT")
            print("  d. ALL models")
            model_choice = input("Choice: ").strip().lower()
            
            if model_choice == 'a':
                step_3_train_vua('roberta')
            elif model_choice == 'b':
                step_3_train_vua('deberta')
            elif model_choice == 'c':
                step_3_train_vua('bert')
            elif model_choice == 'd':
                step_3_train_all_models()
            else:
                print("Invalid choice")
        elif choice == '4':
            print("\nWhich model to fine-tune?")
            print("  a. RoBERTa")
            print("  b. DeBERTa")
            print("  c. BERT")
            model_choice = input("Choice: ").strip().lower()
            
            if model_choice == 'a':
                step_4_finetune_reddit('roberta')
            elif model_choice == 'b':
                step_4_finetune_reddit('deberta')
            elif model_choice == 'c':
                step_4_finetune_reddit('bert')
        elif choice == '5':
            print("\nWhich model to evaluate?")
            print("  a. RoBERTa")
            print("  b. DeBERTa")
            print("  c. BERT")
            model_choice = input("Choice: ").strip().lower()
            
            if model_choice == 'a':
                step_5_evaluate('roberta')
            elif model_choice == 'b':
                step_5_evaluate('deberta')
            elif model_choice == 'c':
                step_5_evaluate('bert')
        else:
            print("Invalid choice")
        
        again = input("\nRun another step? (y/n): ")
        if again.lower() != 'y':
            break

def main():
    parser = argparse.ArgumentParser(description='Reddit Figurative Language Detection')
    parser.add_argument('--step', type=int, choices=[1,2,3,4,5], help='Run specific step')
    parser.add_argument('--scrape', action='store_true', help='Scrape Reddit data')
    parser.add_argument('--train', action='store_true', help='Train all models (RoBERTa, DeBERTa, BERT)')
    parser.add_argument('--train-roberta', action='store_true', help='Train only RoBERTa')
    parser.add_argument('--train-deberta', action='store_true', help='Train only DeBERTa')
    parser.add_argument('--train-bert', action='store_true', help='Train only BERT')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune on Reddit')
    parser.add_argument('--model', type=str, choices=['roberta', 'deberta', 'bert'], 
                       default='roberta', help='Model to use (default: roberta)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    
    args = parser.parse_args()
    
    if args.scrape or args.step == 1:
        step_1_scrape()
    elif args.step == 2:
        step_2_prepare_vua()
    elif args.train or args.step == 3:
        step_3_train_all_models()
    elif args.train_roberta:
        step_3_train_vua('roberta')
    elif args.train_deberta:
        step_3_train_vua('deberta')
    elif args.train_bert:
        step_3_train_vua('bert')
    elif args.finetune or args.step == 4:
        step_4_finetune_reddit(args.model)
    elif args.evaluate or args.step == 5:
        step_5_evaluate(args.model)
    else:
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)