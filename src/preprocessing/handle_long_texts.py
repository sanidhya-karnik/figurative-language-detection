"""
Handle long texts for transformer models with 512 token limit

This module provides utilities to handle texts longer than the model's
maximum sequence length (512 tokens for RoBERTa/DeBERTa).
"""

import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
from typing import List, Dict, Tuple
import warnings

class TextLengthHandler:
    """
    Handle texts that exceed transformer model token limits
    """
    
    def __init__(self, model_name='roberta-base', max_length=512):
        """
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length (512 for RoBERTa/DeBERTa)
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        # Account for special tokens [CLS] and [SEP]
        self.max_content_length = max_length - 2
    
    def get_token_count(self, text: str) -> int:
        """Get number of tokens for a text"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def analyze_dataset_lengths(self, df: pd.DataFrame, text_column='text'):
        """
        Analyze text lengths in dataset
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
        
        Returns:
            Dictionary with statistics
        """
        print("\n" + "="*70)
        print("TEXT LENGTH ANALYSIS")
        print("="*70)
        
        # Calculate token counts
        df['token_count'] = df[text_column].apply(self.get_token_count)
        df['char_count'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()
        
        stats = {
            'total_samples': len(df),
            'tokens': {
                'mean': df['token_count'].mean(),
                'median': df['token_count'].median(),
                'max': df['token_count'].max(),
                'min': df['token_count'].min(),
                'std': df['token_count'].std()
            },
            'chars': {
                'mean': df['char_count'].mean(),
                'median': df['char_count'].median(),
                'max': df['char_count'].max()
            },
            'words': {
                'mean': df['word_count'].mean(),
                'median': df['word_count'].median(),
                'max': df['word_count'].max()
            }
        }
        
        # Count texts exceeding limit
        over_limit = df['token_count'] > self.max_content_length
        stats['over_limit'] = {
            'count': over_limit.sum(),
            'percentage': (over_limit.sum() / len(df)) * 100
        }
        
        # Print statistics
        print(f"\nDataset: {len(df)} samples")
        print(f"\nToken statistics:")
        print(f"  Mean: {stats['tokens']['mean']:.1f} tokens")
        print(f"  Median: {stats['tokens']['median']:.1f} tokens")
        print(f"  Max: {stats['tokens']['max']} tokens")
        print(f"  Std: {stats['tokens']['std']:.1f} tokens")
        
        print(f"\nTexts exceeding {self.max_content_length} tokens:")
        print(f"  Count: {stats['over_limit']['count']} ({stats['over_limit']['percentage']:.1f}%)")
        
        # Distribution
        print(f"\nToken count distribution:")
        bins = [0, 128, 256, 384, 512, 1024, 2048, float('inf')]
        labels = ['0-128', '128-256', '256-384', '384-512', '512-1024', '1024-2048', '2048+']
        df['token_bin'] = pd.cut(df['token_count'], bins=bins, labels=labels)
        print(df['token_bin'].value_counts().sort_index())
        
        return stats
    
    def truncate_simple(self, text: str, strategy='head') -> str:
        """
        Simple truncation strategy
        
        Args:
            text: Input text
            strategy: 'head' (keep beginning), 'tail' (keep end), 'middle' (keep middle)
        
        Returns:
            Truncated text
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_content_length:
            return text
        
        if strategy == 'head':
            # Keep first N tokens
            truncated_tokens = tokens[:self.max_content_length]
        elif strategy == 'tail':
            # Keep last N tokens
            truncated_tokens = tokens[-self.max_content_length:]
        elif strategy == 'middle':
            # Keep middle tokens
            start_idx = (len(tokens) - self.max_content_length) // 2
            truncated_tokens = tokens[start_idx:start_idx + self.max_content_length]
        else:
            truncated_tokens = tokens[:self.max_content_length]
        
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def chunk_with_overlap(self, text: str, overlap=50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            overlap: Number of overlapping tokens between chunks
        
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_content_length:
            return [text]
        
        chunks = []
        stride = self.max_content_length - overlap
        
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.max_content_length]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Stop if we've covered all tokens
            if i + self.max_content_length >= len(tokens):
                break
        
        return chunks
    
    def smart_truncate(self, text: str) -> str:
        """
        Intelligent truncation that tries to keep complete sentences
        
        Args:
            text: Input text
        
        Returns:
            Truncated text maintaining sentence boundaries
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_content_length:
            return text
        
        # Try to find a good breaking point (sentence boundary)
        # Split into sentences
        sentences = text.split('. ')
        
        current_text = ""
        for sentence in sentences:
            test_text = current_text + sentence + ". "
            test_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
            
            if len(test_tokens) > self.max_content_length:
                break
            
            current_text = test_text
        
        # If we got at least some content, return it
        if current_text:
            return current_text.strip()
        
        # Otherwise, fall back to simple truncation
        return self.truncate_simple(text, strategy='head')
    
    def process_dataset(self, df: pd.DataFrame, 
                       text_column='text',
                       strategy='smart_truncate',
                       add_metadata=True) -> pd.DataFrame:
        """
        Process entire dataset to handle long texts
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            strategy: 'simple', 'smart_truncate', 'chunk' (for later use)
            add_metadata: Whether to add metadata about truncation
        
        Returns:
            Processed DataFrame
        """
        print("\n" + "="*70)
        print(f"PROCESSING DATASET - Strategy: {strategy}")
        print("="*70)
        
        # Analyze before processing
        stats_before = self.analyze_dataset_lengths(df, text_column)
        
        df_processed = df.copy()
        
        if add_metadata:
            df_processed['original_length'] = df_processed[text_column].apply(
                self.get_token_count
            )
            df_processed['was_truncated'] = df_processed['original_length'] > self.max_content_length
        
        # Apply truncation
        print(f"\nApplying {strategy} truncation...")
        
        if strategy == 'simple':
            df_processed[text_column] = df_processed[text_column].apply(
                lambda x: self.truncate_simple(x, strategy='head')
            )
        elif strategy == 'smart_truncate':
            df_processed[text_column] = df_processed[text_column].apply(
                self.smart_truncate
            )
        else:
            warnings.warn(f"Unknown strategy: {strategy}, using smart_truncate")
            df_processed[text_column] = df_processed[text_column].apply(
                self.smart_truncate
            )
        
        # Analyze after processing
        print(f"\n{'='*70}")
        print("AFTER TRUNCATION")
        stats_after = self.analyze_dataset_lengths(df_processed, text_column)
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Samples truncated: {df_processed['was_truncated'].sum() if add_metadata else 'N/A'}")
        print(f"Max tokens before: {stats_before['tokens']['max']}")
        print(f"Max tokens after: {stats_after['tokens']['max']}")
        print(f"Mean tokens before: {stats_before['tokens']['mean']:.1f}")
        print(f"Mean tokens after: {stats_after['tokens']['mean']:.1f}")
        
        return df_processed


def process_reddit_data(input_file, output_file):
    """
    Process Reddit data with long texts
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
    """
    print("="*70)
    print("PROCESSING REDDIT DATA")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"\nLoaded {len(df)} samples from {input_file}")
    
    # Initialize handler
    handler = TextLengthHandler(model_name='roberta-base', max_length=512)
    
    # Process with smart truncation
    df_processed = handler.process_dataset(
        df,
        text_column='text',
        strategy='smart_truncate',
        add_metadata=True
    )
    
    # Save
    df_processed.to_csv(output_file, index=False)
    print(f"\n✓ Saved processed data to {output_file}")
    
    # Save statistics
    stats_file = output_file.replace('.csv', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("TEXT LENGTH PROCESSING STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Total samples: {len(df_processed)}\n")
        f.write(f"Truncated samples: {df_processed['was_truncated'].sum()}\n")
        f.write(f"Truncation rate: {df_processed['was_truncated'].mean()*100:.1f}%\n")
    
    print(f"✓ Saved statistics to {stats_file}")


def analyze_all_data():
    """
    Analyze all collected data to understand text lengths
    """
    import glob
    
    print("="*70)
    print("ANALYZING ALL COLLECTED DATA")
    print("="*70)
    
    handler = TextLengthHandler()
    
    # Find all CSV files in raw data
    data_files = glob.glob('data/raw/*.csv')
    
    if not data_files:
        print("\nNo data files found in data/raw/")
        return
    
    all_stats = {}
    
    for filepath in data_files:
        print(f"\n{'='*70}")
        print(f"File: {filepath}")
        print(f"{'='*70}")
        
        try:
            df = pd.read_csv(filepath)
            
            if 'text' not in df.columns:
                print("  Skipping - no 'text' column found")
                continue
            
            stats = handler.analyze_dataset_lengths(df)
            all_stats[filepath] = stats
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary across all files
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for filepath, stats in all_stats.items():
        filename = filepath.split('/')[-1]
        print(f"\n{filename}:")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Mean tokens: {stats['tokens']['mean']:.1f}")
        print(f"  Over limit: {stats['over_limit']['percentage']:.1f}%")


def main():
    """Main function for testing"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            analyze_all_data()
        elif sys.argv[1] == 'process' and len(sys.argv) == 4:
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            process_reddit_data(input_file, output_file)
        else:
            print("Usage:")
            print("  python handle_long_texts.py analyze")
            print("  python handle_long_texts.py process <input.csv> <output.csv>")
    else:
        # Run analysis on sample data
        print("Running in demo mode...")
        analyze_all_data()


if __name__ == "__main__":
    main()