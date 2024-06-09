import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import datasets
import random
import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig


config = LongformerConfig()
# config.attention_mode = 'tvm'
train_data, test_data = datasets.load_dataset(
    'imdb', split=['train', 'test'], cache_dir='data/')
print(config)
print("train_data: ", train_data)
print("test_data: ", test_data)


max_length = 2048
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                            gradient_checkpointing=False,
                                                            attention_window=512)
tokenizer = LongformerTokenizerFast.from_pretrained(
    'allenai/longformer-base-4096', max_length=max_length)


def tokenization(batched_text):
    """
    This function will tokenize the text and return the relevant inputs for the model.
    LongformerForSequenceClassification sets the global attention to the <CLS> token by default,
    so there is no need to further modify the inputs.
    """
    return tokenizer(batched_text['text'], padding='max_length', truncation=True, max_length=max_length)


train_data = train_data.map(
    tokenization, batched=True, batch_size=len(train_data))
test_data = test_data.map(tokenization, batched=True,
                          batch_size=len(test_data))
print("Max Length: ", len(train_data['input_ids'][0]))
train_data.set_format(
    'torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm=False,
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=4,
    fp16=True,
    logging_dir='./logs',
    dataloader_num_workers=0,
    report_to=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train()

trainer.save_model('./model/longformer_imdb')

model = LongformerForSequenceClassification.from_pretrained(
    './model/longformer_imdb')
result = trainer.evaluate()
print(result)
