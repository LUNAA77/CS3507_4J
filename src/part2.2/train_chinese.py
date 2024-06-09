import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, set_seed, GPT2Tokenizer
from tqdm import tqdm
import time

device = torch.device('cuda')

def load_local_dataset(path):
    with open(os.path.join(path, 'chat1.jsonl')) as f:
        data = json.load(f)
    # data = data[:split]
    dataset = CustomDataset(data)
    return dataset

bert_tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
bert_tokenizer.eos_token = bert_tokenizer.sep_token
bert_tokenizer.eos_token_id = bert_tokenizer.sep_token_id

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = bert_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["prompt_prefix"]
        output_text = item["first_answer"]
        input_text = str(input_text)
        output_text = str(output_text)

        full_text = input_text + output_text
        encodings = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
        input_ids = encodings['input_ids'].squeeze()
        
        input_part = self.tokenizer(input_text, add_special_tokens=False)['input_ids']
        output_part = self.tokenizer(output_text, add_special_tokens=False)['input_ids']
                
        labels = torch.full_like(input_ids, -100)

        input_part_length = min(len(input_part), 1024)
        output_part_length = min(len(output_part), 1024 - input_part_length)

        labels[input_part_length:input_part_length + output_part_length] = torch.tensor(output_part[:output_part_length])

        if input_part_length + output_part_length > 1024:
            print("Warning: input plus output length exceeds max_length.")
        
        return {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels
        }


dataset = load_local_dataset('../data/chat')

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)

class SFTTrainer:
    def __init__(self, model, train_loader, lr=1e-5, warmup_steps=100, gradient_accumulation_steps=2, max_grad_norm=1.0, save_path='../models/chinese'):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (warmup_steps + 1)))
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = next(model.parameters()).device
        self.save_path = save_path
    
    def save_model(self, epoch):
        save_directory = os.path.join(self.save_path, f'epochnew_{epoch + 1}')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        model_save_path = os.path.join(save_directory, 'pytorch_model.bin')
        self.model.save_pretrained(save_directory)
        print(f"Model saved to {model_save_path}")

    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0
            batch_nums = 0
            print(f"epoch{epoch+1}:")
            self.model.train()
            for batch in tqdm(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Encountered NaN or Inf loss. Skipping this batch.")
                    continue
                loss.backward()
                total_loss += loss.detach().data
                batch_nums += 1
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs} completed.")
            print("avg_loss:", total_loss / batch_nums)
            self.save_model(epoch)
            

trainer = SFTTrainer(model, train_loader)
trainer.train(epochs=1)
