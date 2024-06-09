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
        # 获取输入嵌入
        inputs_embeds = self.transformer.wte(input_ids)
        
        # 获取设备
        device = input_ids.device
        
        # 应用 ROPE
        seq_len = input_ids.shape[1]
        inputs_embeds = rope(inputs_embeds, seq_len, device)
        
        # 调用父类的 forward 方法
        outputs = super().forward(inputs_embeds=inputs_embeds, **kwargs)
        
        # 如果提供了 labels，则计算损失
        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model=GPT2WithROPE.from_pretrained('../models/rope').to(device)
model.eval()

def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    # attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            # attention_mask=attention_mask,
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "When I was a child, I used to"
print(generate_text(text, max_length=100))


