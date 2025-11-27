"""
Process VUA Metaphor Corpus for figurative language detection

The VUA corpus has word-level annotations. We convert to sentence-level:
- If ANY word in a sentence is labeled as metaphor (label=1), the sentence is "metaphor"
- If ALL words are literal (label=0), the sentence is "literal"
"""

import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

class VUAProcessor:
    """Process VUA Metaphor Corpus data"""
    
    def __init__(self, vua_base_path='data/vua_archive'):
        """
        Args:
            vua_base_path: Base path to VUA archive folders
        """
        self.vua_base_path = Path(vua_base_path)
        self.output_path = Path('data/processed')
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_tsv_file(self, filepath):
        """
        Load a VUA TSV file
        
        Expected columns: index, label, sentence, pos, w_index, target, word_sense, definition
        """
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
            print(f"  Loaded {len(df)} rows from {filepath.name}")
            print(f"  Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            return None
    
    def process_word_level_to_sentence(self, df):
        """
        Convert word-level metaphor annotations to sentence-level
        
        Args:
            df: DataFrame with word-level annotations
                Expected columns: 'sentence' (text), 'label' (0 or 1 for each word)
        
        Returns:
            DataFrame with sentence-level labels
        """
        if 'sentence' not in df.columns or 'label' not in df.columns:
            print(f"  Warning: Required columns not found. Available: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Group by sentence
        # Each row is a word, we need to group words by their sentence
        sentences = []
        
        # Get unique sentences (assuming 'sentence' column contains full sentence text)
        for sentence_text in df['sentence'].unique():
            # Get all words for this sentence
            sentence_words = df[df['sentence'] == sentence_text]
            
            # Check if ANY word is labeled as metaphor (label == 1)
            has_metaphor = (sentence_words['label'] == 1).any()
            
            # Count metaphorical words
            num_metaphors = (sentence_words['label'] == 1).sum()
            
            sentences.append({
                'text': sentence_text,
                'label': 'metaphor' if has_metaphor else 'literal',
                'num_words': len(sentence_words),
                'num_metaphors': num_metaphors,
                'metaphor_ratio': num_metaphors / len(sentence_words) if len(sentence_words) > 0 else 0
            })
        
        result_df = pd.DataFrame(sentences)
        
        print(f"  Converted {len(df)} words to {len(result_df)} sentences")
        print(f"  Metaphor sentences: {(result_df['label'] == 'metaphor').sum()}")
        print(f"  Literal sentences: {(result_df['label'] == 'literal').sum()}")
        
        return result_df
    
    def process_vua_folder(self, folder_name):
        """
        Process a single VUA folder (e.g., VUA18, VUA20, etc.)
        
        Args:
            folder_name: Name of VUA folder
        
        Returns:
            Dictionary with train and test DataFrames
        """
        folder_path = self.vua_base_path / folder_name
        
        if not folder_path.exists():
            print(f"  Folder not found: {folder_path}")
            return None
        
        print(f"\nProcessing {folder_name}...")
        
        train_file = folder_path / 'train.tsv'
        test_file = folder_path / 'test.tsv'
        
        result = {}
        
        # Process train file
        if train_file.exists():
            print(f"  Loading train data...")
            train_df = self.load_tsv_file(train_file)
            if train_df is not None and not train_df.empty:
                result['train'] = self.process_word_level_to_sentence(train_df)
                if not result['train'].empty:
                    print(f"  ✓ Processed {len(result['train'])} training sentences")
        
        # Process test file
        if test_file.exists():
            print(f"  Loading test data...")
            test_df = self.load_tsv_file(test_file)
            if test_df is not None and not test_df.empty:
                result['test'] = self.process_word_level_to_sentence(test_df)
                if not result['test'].empty:
                    print(f"  ✓ Processed {len(result['test'])} test sentences")
        
        return result if result else None
    
    def process_vua_pos_folder(self, folder_name='VUA18_pos'):
        """
        Process VUA18_pos folder with separate POS subfolders
        
        Args:
            folder_name: Name of VUA POS folder
        
        Returns:
            Dictionary with train and test DataFrames combined from all POS
        """
        folder_path = self.vua_base_path / folder_name
        
        if not folder_path.exists():
            print(f"  Folder not found: {folder_path}")
            return None
        
        print(f"\nProcessing {folder_name} (POS-specific)...")
        
        pos_folders = ['adj', 'adv', 'noun', 'verb']
        all_train = []
        all_test = []
        
        for pos in pos_folders:
            pos_path = folder_path / pos
            if not pos_path.exists():
                continue
            
            print(f"\n  Processing {pos}...")
            
            # Train
            train_file = pos_path / 'train.tsv'
            if train_file.exists():
                train_df = self.load_tsv_file(train_file)
                if train_df is not None and not train_df.empty:
                    processed = self.process_word_level_to_sentence(train_df)
                    if not processed.empty:
                        processed['pos'] = pos
                        all_train.append(processed)
            
            # Test
            test_file = pos_path / 'test.tsv'
            if test_file.exists():
                test_df = self.load_tsv_file(test_file)
                if test_df is not None and not test_df.empty:
                    processed = self.process_word_level_to_sentence(test_df)
                    if not processed.empty:
                        processed['pos'] = pos
                        all_test.append(processed)
        
        result = {}
        if all_train:
            result['train'] = pd.concat(all_train, ignore_index=True)
            print(f"\n  ✓ Combined {len(result['train'])} training sentences from all POS")
        
        if all_test:
            result['test'] = pd.concat(all_test, ignore_index=True)
            print(f"  ✓ Combined {len(result['test'])} test sentences from all POS")
        
        return result if result else None
    
    def create_validation_split(self, train_df, val_size=0.1):
        """Create validation split from training data"""
        train, val = train_test_split(
            train_df,
            test_size=val_size,
            stratify=train_df['label'],
            random_state=42
        )
        return train, val
    
    def process_all_vua_data(self):
        """
        Process all VUA data and create unified train/val/test splits
        """
        print("="*70)
        print("PROCESSING VUA METAPHOR CORPUS")
        print("="*70)
        
        all_datasets = {}
        
        # List of VUA folders to process
        vua_folders = [
            'VUA_MPD',
            'VUA18',
            'VUA18-',
            'VUA20',
            'VUA20-',
            'VUAverb',
            'VUAverb-'
        ]
        
        # Process regular VUA folders
        for folder in vua_folders:
            result = self.process_vua_folder(folder)
            if result:
                all_datasets[folder] = result
        
        # Process VUA18_pos separately
        vua_pos_result = self.process_vua_pos_folder('VUA18_pos')
        if vua_pos_result:
            all_datasets['VUA18_pos'] = vua_pos_result
        
        if not all_datasets:
            print("\n✗ No VUA data processed successfully!")
            return
        
        # Combine all datasets
        print("\n" + "="*70)
        print("COMBINING ALL VUA DATASETS")
        print("="*70)
        
        all_train = []
        all_test = []
        
        for name, dataset in all_datasets.items():
            if 'train' in dataset and not dataset['train'].empty:
                dataset['train']['source'] = name
                all_train.append(dataset['train'])
            if 'test' in dataset and not dataset['test'].empty:
                dataset['test']['source'] = name
                all_test.append(dataset['test'])
        
        if not all_train or not all_test:
            print("\n✗ No data to combine!")
            return
        
        # Combine
        combined_train = pd.concat(all_train, ignore_index=True)
        combined_test = pd.concat(all_test, ignore_index=True)
        
        # Remove duplicates based on text
        print(f"\nBefore deduplication:")
        print(f"  Train: {len(combined_train)}")
        print(f"  Test: {len(combined_test)}")
        
        combined_train = combined_train.drop_duplicates(subset=['text'], keep='first')
        combined_test = combined_test.drop_duplicates(subset=['text'], keep='first')
        
        print(f"\nAfter deduplication:")
        print(f"  Train: {len(combined_train)}")
        print(f"  Test: {len(combined_test)}")
        
        # Label distribution
        print(f"\nTraining label distribution:")
        print(combined_train['label'].value_counts())
        print(f"\nTest label distribution:")
        print(combined_test['label'].value_counts())
        
        # Create validation split
        train_final, val_final = self.create_validation_split(combined_train)
        
        print(f"\nFinal splits:")
        print(f"  Train: {len(train_final)}")
        print(f"  Val: {len(val_final)}")
        print(f"  Test: {len(combined_test)}")
        
        # Save processed data
        print("\n" + "="*70)
        print("SAVING PROCESSED DATA")
        print("="*70)
        
        # Save as CSV (only text and label for training)
        train_final[['text', 'label']].to_csv(
            self.output_path / 'vua_train.csv',
            index=False
        )
        val_final[['text', 'label']].to_csv(
            self.output_path / 'vua_val.csv',
            index=False
        )
        combined_test[['text', 'label']].to_csv(
            self.output_path / 'vua_test.csv',
            index=False
        )
        
        print(f"\n✓ Saved train data: {self.output_path / 'vua_train.csv'}")
        print(f"✓ Saved validation data: {self.output_path / 'vua_val.csv'}")
        print(f"✓ Saved test data: {self.output_path / 'vua_test.csv'}")
        
        # Save detailed version with metadata
        train_final.to_csv(
            self.output_path / 'vua_train_detailed.csv',
            index=False
        )
        val_final.to_csv(
            self.output_path / 'vua_val_detailed.csv',
            index=False
        )
        combined_test.to_csv(
            self.output_path / 'vua_test_detailed.csv',
            index=False
        )
        
        print(f"\n✓ Saved detailed versions with metadata")
        
        print("\n" + "="*70)
        print("VUA PROCESSING COMPLETE!")
        print("="*70)
        
        return {
            'train': train_final,
            'val': val_final,
            'test': combined_test
        }


def main():
    """Main processing function"""
    
    # Check if VUA data exists
    vua_path = Path('data/vua_archive')
    
    if not vua_path.exists():
        print("="*70)
        print("ERROR: VUA data not found")
        print("="*70)
        print(f"\nExpected location: {vua_path.absolute()}")
        print("\nPlease ensure your VUA data is in the correct location:")
        print("  data/")
        print("    vua_archive/")
        print("      VUA_MPD/")
        print("      VUA18/")
        print("      VUA18_pos/")
        print("      ... (other VUA folders)")
        return
    
    # Process VUA data
    processor = VUAProcessor(vua_base_path=vua_path)
    results = processor.process_all_vua_data()
    
    if results:
        print("\n✓ VUA data ready for training!")
        print("\nNext steps:")
        print("  1. Train model: python run_reddit.py --train")


if __name__ == "__main__":
    main()