import torch
from torch import nn
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import evaluate
import wandb

from config import Config
from transformer import Transformer
from utils.training_utils import ExponentialMovingAverage, WarmupCosineScheduler, clean_text, tokenize, train, generate, evaluate


if __name__ == "__main__":

    data_paths = {'train': Config.train_data_path, 'test': Config.test_data_path}
    dataset = load_dataset('csv', data_files=data_paths)

    train_loader = DataLoader(dataset['train'], shuffle=True, batch_size=Config.batch_size)
    test_loader = DataLoader(dataset['test'], shuffle=False, batch_size=Config.batch_size)

    print("Data is loaded")

    tokenizers = {
        Config.source_language: AutoTokenizer.from_pretrained(Config.source_tokenizer),
        Config.target_language: AutoTokenizer.from_pretrained(Config.target_tokenizer)
    }

    print("Tokenizers are set")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(
        encoder_vocab_size=len(tokenizers[Config.source_language]), 
        decoder_vocab_size=len(tokenizers[Config.target_language]), 
        hidden_size=Config.hidden_size, 
        n_head=Config.number_of_attention_heads,
        intermediate_size=Config.feed_forward_intermediate_size, 
        encoder_max_len=Config.max_length, 
        decoder_max_len=Config.max_length, 
        n_layers=Config.number_of_blocks, 
        drop_prob=Config.dropout_rate
    ).to(device)
    
    print('Model parameters:', sum(torch.numel(p) for p in model.parameters()))

    ema_model = ExponentialMovingAverage(model.parameters(), decay=Config.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=Config.warmup_steps, max_training_steps=Config.max_training_steps, min_lr=Config.min_learming_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizers[Config.target_language].pad_token_id)

    if Config.use_wandb:
        wandb.init(project=Config.wandb_project_name, name=Config.wanndb_run_name)

    for e in range(Config.number_of_epochs):
        train(
            model,
            ema_model,
            train_loader,
            tokenizers,
            criterion,
            optimizer,
            scheduler,
            device
        )
        
        evaluate(
            model,
            test_loader,
            tokenizers,
            criterion,
            device
        )
        
