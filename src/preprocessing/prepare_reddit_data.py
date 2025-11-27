import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare_reddit_annotations():
    """Split Reddit annotations into train/val/test with binary labels"""
    
    # Load annotations
    annotations_file = Path('data/processed/reddit_annotations.csv')
    
    if not annotations_file.exists():
        print(f"❌ File not found: {annotations_file}")
        return
    
    df = pd.read_csv(annotations_file)
    
    # Convert to binary labels
    df['label_binary'] = df['label'].apply(
        lambda x: 'literal' if x == 'literal' else 'figurative'
    )
    
    print(f"Loaded {len(df)} annotations")
    print(f"\nOriginal label distribution:")
    print(df['label'].value_counts())
    print(f"\nBinary label distribution:")
    print(df['label_binary'].value_counts())
    
    # Split: 60% train, 20% val, 20% test
    # Use binary labels for stratification
    train, temp = train_test_split(
        df, 
        test_size=0.4, 
        stratify=df['label_binary'],  # Changed to binary
        random_state=42
    )
    val, test = train_test_split(
        temp, 
        test_size=0.5, 
        stratify=temp['label_binary'],  # Changed to binary
        random_state=42
    )
    
    print(f"\nSplits:")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    
    print(f"\nTrain label distribution:")
    print(train['label_binary'].value_counts())
    
    # Save with binary labels (only text and label_binary)
    train[['text', 'label_binary']].rename(columns={'label_binary': 'label'}).to_csv(
        'data/processed/reddit_train.csv', index=False
    )
    val[['text', 'label_binary']].rename(columns={'label_binary': 'label'}).to_csv(
        'data/processed/reddit_val.csv', index=False
    )
    test[['text', 'label_binary']].rename(columns={'label_binary': 'label'}).to_csv(
        'data/processed/reddit_test.csv', index=False
    )
    
    print("\n✓ Saved binary splits to data/processed/")
    print("  - reddit_train.csv (literal vs figurative)")
    print("  - reddit_val.csv")
    print("  - reddit_test.csv")

if __name__ == "__main__":
    prepare_reddit_annotations()