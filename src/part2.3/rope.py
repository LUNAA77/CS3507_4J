import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def rope(x, seq_len, device):
    dim = x.shape[-1]
    half_dim = dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (freqs / half_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = pos[:, None] * freqs[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return x * torch.cos(angles) + torch.roll(x, 1, dims=-1) * torch.sin(angles)


class GPT2WithROPE(GPT2LMHeadModel):
    def forward(self, input_ids=None, labels=None, **kwargs):
        inputs_embeds = self.transformer.wte(input_ids)
        device = input_ids.device
        seq_len = input_ids.shape[1]
        inputs_embeds = rope(inputs_embeds, seq_len, device)
        outputs = super().forward(inputs_embeds=inputs_embeds, **kwargs)
        
        # # 如果提供了 labels，则计算损失
        # if labels is not None:
        #     shift_logits = outputs.logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = torch.nn.CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #     return CausalLMOutputWithCrossAttentions(
        #         loss=loss,
        #         logits=outputs.logits,
        #         past_key_values=outputs.past_key_values,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #         cross_attentions=outputs.cross_attentions,
        #     )

        return outputs

# 实例化模型并加载预训练权重
model = GPT2WithROPE.from_pretrained('gpt2')

# 加载数据集和 tokenizer
dataset = load_dataset('ptb_text_only')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=1024)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids'])

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy='epoch',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'labels': labels}

def move_batch_to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        return [move_batch_to_device(batch, device) for batch in dataloader]

    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = super().get_eval_dataloader(eval_dataset)
        return [move_batch_to_device(batch, device) for batch in dataloader]

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=collate_fn,
)


trainer.train()

# 保存模型
model.save_pretrained('../models/ropeee')

# 评估
results = trainer.evaluate()
print(results)
