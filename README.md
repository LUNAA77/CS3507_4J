## Install Dependencies

To install dependencies for experiments, run:

```bash
pip install -r requirements.txt
```

In the following guidance, you may need to modify some file paths in the code before running it.

## 1.1 Sentiment Analysis

To run sentiment analysis:

1. Run wash_data.csv
2. Run the command below in the terminal

```bash 
python ./src/part1.1/sentiment_analysis.py
```

## 1.2 Code Generation

To run code generation, run

```bash
python ./src/part1.2/train_sft.py
```

## 1.3 Summarization

To run summarization, run

```
python ./src/part1.3/sum.py
```

to train the model, run

```
python ./src/part1.3/inference.py
```

to test the model.

## 2.1 Text classification

To run text classification, run

```bash
python ./src/part2.1/Classification_Longformer.py
python ./src/part2.1/Classification_Roberta.py
```

It may take several hours to finish training.

## 2.2 Apply on Chinese

```bash
# train
python ./src/part2.2/train_chinese.py

# test
python ./src/part2.2/chinese.py
```

## 2.3 Position Encoding

```bash
# train
python ./src/part2.3/rope.py

# test
python ./src/part2.3/rope_eval.py
```
