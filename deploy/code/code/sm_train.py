#%%
import os
import sys
import argparse
import logging

import pandas as pd
import numpy as np
import deepmatcher
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepmatcher.data.custom_dataset import DeepMatcherDataset
from deepmatcher.data.encoder.text_encoders import HFTextEncoder
from deepmatcher.models.core import MatchingModel
from deepmatcher.optim import Optimizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_dataloader(data_df, label_column, text_cols, image_col, tokenizer, batch_size, shuffle, num_workers):
    dataset = DeepMatcherDataset(
        data_df=data_df,
        label_col=label_column,
        text_cols=text_cols,
        image_col=image_col,
        tokenizer=tokenizer
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate_fn)

def train(args):
    dataset_dir = args.data_dir
    train_df = pd.read_json(os.path.join(dataset_dir, "train.json.gz"), compression='gzip', lines=True)
    val_df = pd.read_json(os.path.join(dataset_dir, "val.json.gz"), compression='gzip', lines=True)
    test_df = pd.read_json(os.path.join(dataset_dir, "test.json.gz"), compression='gzip', lines=True)
    
    text_encoder = HFTextEncoder(args.text_encoder, trainable=args.trainable, seqtovec=args.seq_to_vec)
    
    label_column = args.label_col
    text_cols = args.text_attrs
    image_col = args.image_attr
    tokenizer = text_encoder.tokenizer

    train_dataloader = get_dataloader(train_df, label_column, text_cols, image_col, tokenizer, args.batch_size, True, args.num_workers)
    val_dataloader = get_dataloader(val_df, label_column, text_cols, image_col, tokenizer, args.batch_size, False, args.num_workers)
    test_dataloader = get_dataloader(test_df, label_column, text_cols, image_col, tokenizer, args.batch_size, False, args.num_workers)

    model = MatchingModel(
        text_encoder=text_encoder,
        # image_encoder=image_encoder,
        attr_summarizer=None,
        attr_condense_factor=args.attr_condense_factor,
        attr_comparator=args.attr_comparator,
        attr_merge=args.attr_merge,
        classifier=args.classifier,
        hidden_size=args.hidden_size,
        text_attrs=text_cols,
        image_attr=image_col
    )
    
    logger.info('intializing model')
    model.initialize(next(iter(train_dataloader))['attrs'])
    logger.info(f"intialized model:\n{model}")
    
    # optimizer = Optimizer(lr=1e-3, lr_decay=1e-1, start_decay_at=3)
    
    logger.info('Running Model Training')
    model.run_train(
        train_dataloader,
        val_dataloader,
        best_save_path=os.path.join(args.model_dir, "model.pth"),
        epochs=args.epochs
        )

    logger.info('Evaluating Model')
    eval_f1_score = model.run_eval(test_dataloader)
    logger.info('Score on test set: %s', eval_f1_score)
    return model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--text_encoder', type=str, default='albert-base-v2')
    parser.add_argument('--trainable', type=bool, default=False)
    parser.add_argument('--seq_to_vec', type=str, default='cls', choices=['cls', 'mean'])

    parser.add_argument('--text_attrs', type=str, nargs="*", default=[])
    parser.add_argument('--image_attr', type=str, default='')
    parser.add_argument('--label_col', type=str, default='label')

    parser.add_argument('--attr_condense_factor', type=str, default='auto')
    parser.add_argument('--attr_comparator', type=str, default='concat-abs-diff')
    parser.add_argument('--attr_merge', type=str, default='concat')
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--classifier', type=str, default='2-layer-highway')

    # Data, model, and output directories
    # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--embedding-dir', type=str, default=os.environ['SM_CHANNEL_EMBEDDING'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    # Training params
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=1)

    args, _ = parser.parse_known_args()
    logger.debug('Args: %s', args)
    model = train(args)