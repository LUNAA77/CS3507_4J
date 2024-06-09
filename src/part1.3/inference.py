import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device:", device)

original_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
original_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=original_tokenizer.eos_token_id).to(device)
if original_tokenizer.pad_token is None:
    original_tokenizer.pad_token = original_tokenizer.eos_token
    original_tokenizer.add_special_tokens({'pad_token': original_tokenizer.eos_token})
original_model.resize_token_embeddings(len(original_tokenizer))
original_model.eval()


finetuned_model_path = '/home/shuochen/CS3507/models/sum_3'
# finetuned_tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_path)
finetuned_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
finetuned_model = GPT2LMHeadModel.from_pretrained(finetuned_model_path, pad_token_id=finetuned_tokenizer.eos_token_id).to(device)
finetuned_model.eval()

# finetuned_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
# finetuned_tokenizer = tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# finetuned_model.resize_token_embeddings(len(finetuned_tokenizer))
# finetuned_model.load_state_dict(torch.load('./models/gpt2_medium_code_alpaca_4.pt'))
# finetuned_model.eval()

text = "你好，请问你是gpt2吗？"

original_inputs = original_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
original_inputs = {k: v.to(device) for k, v in original_inputs.items()}
finetuned_inputs = finetuned_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
finetuned_inputs = {k: v.to(device) for k, v in finetuned_inputs.items()}


# total_max_length = len(original_inputs['input_ids'][0]) + 150
total_max_length = 512
original_outputs = original_model.generate(
    original_inputs['input_ids'], 
    attention_mask=original_inputs['attention_mask'],
    max_length=total_max_length, 
    num_return_sequences=1,
    pad_token_id=original_tokenizer.pad_token_id,
    temperature=0.3,
    # top_k=40,
    # top_p=0.75,
    repetition_penalty=1.2,
    early_stopping=True, 
    length_penalty=2.0
)
finetuned_outputs = finetuned_model.generate(
    finetuned_inputs['input_ids'], 
    attention_mask=finetuned_inputs['attention_mask'],
    max_length=total_max_length, 
    num_return_sequences=1,
    pad_token_id=finetuned_tokenizer.pad_token_id,
    temperature=0.3,
    # top_k=40,
    # top_p=0.75,
    repetition_penalty=1.2,
    early_stopping=True, 
    length_penalty=2.0
)


original_result = original_tokenizer.decode(original_outputs[0], skip_special_tokens=True)
finetuned_result = finetuned_tokenizer.decode(finetuned_outputs[0], skip_special_tokens=True)

print("\nOriginal Model Output:")
print(original_result)

print("\nFine-tuned Model Output:")
print(finetuned_result)