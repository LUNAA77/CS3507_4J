import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda')

def load_local_dataset(path):
    with open(os.path.join(path, 'code_alpaca_20k.json')) as f:
        data = json.load(f)
    dataset = CustomDataset(data)
    return dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]

        full_text = instruction + input_text + output_text
        encodings = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        input_ids = encodings['input_ids'].squeeze()
        
        input_part = self.tokenizer(instruction + input_text, add_special_tokens=False)['input_ids']
        output_part = self.tokenizer(output_text, add_special_tokens=False)['input_ids']
                
        labels = torch.full_like(input_ids, -100)

        input_part_length = min(len(input_part), 256)
        output_part_length = min(len(output_part), 256 - input_part_length)

        labels[input_part_length:input_part_length + output_part_length] = torch.tensor(output_part[:output_part_length])

        if input_part_length + output_part_length > 256:
            print("Warning: input plus output length exceeds max_length.")
        
        return {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels
        }


dataset = load_local_dataset('./data')

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

class SFTTrainer:
    def __init__(self, model, train_loader, lr=1e-5, warmup_steps=100, gradient_accumulation_steps=2, max_grad_norm=1.0, save_path='./models'):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (warmup_steps + 1)))
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = next(model.parameters()).device
        self.save_path = save_path
    
    def save_model(self, epoch):
        save_directory = os.path.join(self.save_path, f'epoch_{epoch + 1}')
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
trainer.train(epochs=10)
