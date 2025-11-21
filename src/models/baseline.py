"""
Baseline model definitions for RoBERTa, DeBERTa, and BERT

This module provides classifier classes for three transformer models:
- RoBERTa (BaselineClassifier)
- BERT (BertClassifier)
- DeBERTa (DeBERTaClassifier)

All classifiers are configured for binary classification tasks.
"""

import torch
import torch.nn as nn
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizer, 
    RobertaConfig,
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer
)


class BaselineClassifier:
    """
    RoBERTa classifier for binary classification
    (metaphor vs literal OR irony vs literal)
    """
    
    def __init__(self, model_name='roberta-base', num_labels=2, task_name='metaphor'):
        """
        Args:
            model_name: 'roberta-base' or 'roberta-large'
            num_labels: 2 for binary classification
            task_name: 'metaphor' or 'irony'
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.task_name = task_name
        
        # Label mapping
        self.label2id = {'literal': 0, task_name: 1}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize
        print(f"Loading {model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        print(f"✓ {model_name} loaded successfully")
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer


class BertClassifier:
    """
    BERT classifier for binary classification
    """
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, task_name='metaphor'):
        """
        Args:
            model_name: 'bert-base-uncased' or 'bert-large-uncased'
            num_labels: 2 for binary classification
            task_name: 'metaphor' or 'irony'
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.task_name = task_name
        
        # Label mapping
        self.label2id = {'literal': 0, task_name: 1}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize
        print(f"Loading {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        print(f"✓ {model_name} loaded successfully")
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer


class DeBERTaClassifier:
    """
    DeBERTa-v3 classifier for binary classification
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=2, task_name='metaphor'):
        """
        Args:
            model_name: 'microsoft/deberta-v3-base' or 'microsoft/deberta-v3-large'
            num_labels: 2 for binary classification
            task_name: 'metaphor' or 'irony'
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.task_name = task_name
        
        # Label mapping
        self.label2id = {'literal': 0, task_name: 1}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize
        print(f"Loading {model_name}...")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        print(f"✓ {model_name} loaded successfully")
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer