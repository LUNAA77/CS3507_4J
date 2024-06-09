import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, set_seed, GPT2Tokenizer
from datasets import load_dataset
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
set_seed(42)

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.eos_token = tokenizer.sep_token
tokenizer.eos_token_id = tokenizer.sep_token_id
eos_token_id = tokenizer.sep_token_id
 
model = GPT2LMHeadModel.from_pretrained("/home/shuochen/4J/models/chinese/epochnew_1").to(device)
# model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
# generation
model.eval()
text_generator = TextGenerationPipeline(model, tokenizer, device=device)

result = text_generator(
    "你会做数学题吗？",
    max_length=256, 
    do_sample=True, 
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=eos_token_id,
    early_stopping=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)


print(''.join(result[0]['generated_text'].split()))
