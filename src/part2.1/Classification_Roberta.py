import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import datasets
import random
import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


train_data, test_data = datasets.load_dataset('imdb', split =['train', 'test'], cache_dir='data/')
# train_data = train_data.shuffle(seed = 42)
# test_data = test_data.shuffle(seed = 42)
# train_data = train_data.select(range(10000))
# test_data = test_data.select(range(10000))
print("train_data: ", train_data)
print("test_data: ", test_data)

model = RobertaForSequenceClassification.from_pretrained('roberta-base')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)

def tokenization(batched_text):
    return tokenizer(batched_text['text'], padding = True, truncation=True)


train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
# define the training arguments
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 16,    
    per_device_eval_batch_size = 32,
    disable_tqdm = False, 
    warmup_steps = 500,
    weight_decay = 0.01,
    logging_steps = 8,
    fp16 = True,
    logging_dir='./logs',
    dataloader_num_workers = 0,
    report_to =  None
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train()

trainer.save_model('./model/roberta_imdb')

model = RobertaForSequenceClassification.from_pretrained('./model/roberta_imdb')
result = trainer.evaluate()
print(result)