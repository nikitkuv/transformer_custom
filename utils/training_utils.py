import torch
from torch import nn
from torch import Tensor
import numpy as np
import evaluate
from tqdm import tqdm
import wandb
import math

from config import Config


class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.999):
        
        """
        parameters: параметры модели model.parameters()
        decay: параметр lambda, регулирует скорость забывания старой информации
        """
        
        super().__init__()

        self.decay = decay
        self.parameters = [p.clone().detach() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def update(self, parameters):
        
        """
        Обновляет текущие параметры EMA
        parameters: обновленные параметры модели
        """
        
        current_params = [p for p in parameters if p.requires_grad]

        # ema_new = decay * ema_old + (1 - decay) * current_param
        for ema_param, current_param in zip(self.parameters, current_params):
            ema_param.mul_(self.decay).add_(current_param.detach(), alpha=1-self.decay)

    def state_dict(self) -> dict:
        
        """
        Возвращает словарь с текущими параметрами EMA
        """
        
        return {'parameters': self.parameters}

    def copy_to(self, parameters):
        
        """
        Копирует сохраненные параметры в полученные параметры
        parameters: параметры модели для копирования
        """
        
        target_params = [p for p in parameters if p.requires_grad]

        for ema_param, target_param in zip(self.parameters, target_params):
            target_param.data.copy_(ema_param.data)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps=2000, max_training_steps=40000, min_lr=1e-6):
        
        """
        optimizer: оптимизатор
        warmup_steps: число шагов warmup
        max_training_steps: общее число шагов обучения
        min_lr: минимальное значение скорости обучения
        """
        
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_training_steps = max_training_steps
        self.min_lr = min_lr

        self.max_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        
        """
        Делает шаг обновления скорости обучения optimizer.
        """

        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            frac=self.step_count/self.warmup_steps
            new_lr = self.min_lr * (1 - frac) + self.max_lr * frac
        else:
            if self.step_count >= self.max_training_steps:
                new_lr = self.min_lr
            else:
                lr_diff = self.max_lr - self.min_lr
                t_m_tw = self.step_count - self.warmup_steps
                tm_m_tw = self.max_training_steps - self.warmup_steps

                new_lr = self.min_lr + 0.5 * lr_diff * (1 + math.cos(t_m_tw/tm_m_tw * math.pi))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def clean_text(texts):

    new_texts = []
    
    for text in texts:
        cls_pos = text.find('[CLS]')
        text = text[cls_pos + len('[CLS] '):]

        sep_pos = text.find(' [SEP]')
        if sep_pos != -1:
            text = text[:sep_pos]

        new_texts.append(text)

    return new_texts


def tokenize(
    tokenizers, 
    src_texts, 
    trg_texts, 
    device
):

    src_enc = tokenizers[Config.source_language](
        src_texts, 
        padding=True, 
        truncation=True, 
        max_length=Config.max_length, 
        return_tensors='pt'
    )
    
    trg_enc = tokenizers[Config.target_language](
        trg_texts, 
        padding=True, 
        truncation=True, 
        max_length=Config.max_length, 
        return_tensors='pt'
    )

    src_input_ids = src_enc['input_ids'].to(device)
    src_attention_mask = src_enc['attention_mask'].to(device)
    trg_input_ids = trg_enc['input_ids'].to(device)
    trg_attention_mask = trg_enc['attention_mask'].to(device)

    return src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask


def train(
    model,
    ema_model,
    train_loader,
    tokenizers,
    criterion,
    optimizer,
    scheduler,
    device
):

    model.train()

    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        
        src_texts = batch[Config.source_language]
        trg_texts = batch[Config.target_language]

        src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask = tokenize(tokenizers, src_texts, trg_texts, device)

        # смещаем вход для декодера - таргетную последовательность
        shifted_input = trg_input_ids[:, :-1]
        shifted_attention_mask = trg_attention_mask[:, :-1]
        labels = trg_input_ids[:, 1:]

        logits = model(
            src_input_ids, 
            shifted_input, 
            src_attention_mask, 
            shifted_attention_mask
        )

        # переставляем размерности логитов, так как в CrossEntropyLoss предсказания классов должны быть второй размерностью
        logits_reshaped = logits.view(-1, logits.size(-1))
        labels_reshaped = labels.contiguous().view(-1)

        # подсчет лосса
        loss = criterion(logits_reshaped, labels_reshaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        ema_model.update(model.parameters())

        # подсчет точности
        predictions = logits.argmax(-1)
        correct = (predictions == labels)
        non_pad_mask = (labels != tokenizers[Config.target_language].pad_token_id)
        accuracy = correct[non_pad_mask].float().mean().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{accuracy:.4f}',
            'lr': f'{scheduler.get_lr():.2e}'
        })

        if Config.use_wandb:
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "lr": scheduler.get_lr()
            })


@torch.inference_mode()
def generate(
    model, 
    src_text,
    device,
    tokenizers,
    max_len=Config.max_length
):
    
    model.eval()
    
    with torch.no_grad():

        src_enc = tokenizers[Config.source_language](
            src_text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=Config.max_length
        ).to(device)
        
        src_input_ids = src_enc['input_ids']
        src_attention_mask = src_enc['attention_mask']

        trg_input_ids = torch.tensor(
            [[tokenizers[Config.target_language].cls_token_id]], 
            device=device
        )
        
        for _ in range(max_len):

            logits = model(
                src_input_ids, 
                trg_input_ids, 
                src_attention_mask,
                None
            )
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            trg_input_ids = torch.cat(
                [trg_input_ids, next_token.unsqueeze(1)], 
                dim=1
            )
            
            if next_token.item() == tokenizers[Config.target_language].sep_token_id:
                break
        
        return trg_input_ids


def evaluate(
    model, 
    ema_model, 
    test_loader,
    tokenizers,
    criterion,
    device
):

    bleu_score = evaluate.load('bleu')
    
    model.eval()
    accuracies = []
    losses = []
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            
            src_texts = batch[Config.source_language]
            trg_texts = batch[Config.target_language]
            
            src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask = tokenize(tokenizers, src_texts, trg_texts, device)

            # смещаем вход для декодера - таргетную последовательность
            shifted_input = trg_input_ids[:, :-1]
            shifted_attention_mask = trg_attention_mask[:, :-1]
            labels = trg_input_ids[:, 1:]
            
            logits = model(
                src_input_ids, 
                shifted_input, 
                src_attention_mask, 
                shifted_attention_mask
            )
            
            # переставляем размерности логитов, так как в CrossEntropyLoss предсказания классов должны быть второй размерностью
            logits_reshaped = logits.view(-1, logits.size(-1))
            labels_reshaped = labels.contiguous().view(-1)
            
            loss = criterion(logits_reshaped, labels_reshaped)
            losses.append(loss.item())

            predictions = logits.argmax(-1)
            correct = (predictions == labels)
            pad_token_id = torch.tensor(tokenizers[Config.target_language].pad_token_id, device=labels.device)
            non_pad_mask = (labels != pad_token_id)
            accuracy = correct[non_pad_mask].float().mean().item()
            accuracies.append(accuracy)
            
            for i in range(len(src_texts)):
                pred_ids = generate(
                    model, 
                    src_texts[i],
                    device,
                    tokenizers,
                    max_len=Config.max_length, 
                )

                all_preds.extend(clean_text(tokenizers[Config.target_language].batch_decode(pred_ids)))
                all_refs.extend(tokenizers[Config.target_language].batch_decode(trg_input_ids[i].unsqueeze(0), skip_special_tokens=True))
    
    bleu_result = bleu_score.compute(
        predictions=all_preds, 
        references=all_refs
    )

    if Config.use_wandb:
        wandb.log({
            "test_loss": np.mean(losses),
            "test_accuracy": np.mean(accuracies),
            "BLEU": bleu_result['bleu']
        })
    