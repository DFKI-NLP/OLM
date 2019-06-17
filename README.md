# xBert: Occlusion powered by BERT

## Installation

First, clone the repository to your machine and install the requirements with the following command:

```bash
pip install -r requirements.txt
```

## xBert Tasks

### SST2
Dataset (part of [GLUE](https://gluebenchmark.com/tasks)): [Download](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8)

In `xbert_tasks/sst2_sentiment_clf.jsonnet`, change `train_data_path` and `validation_data_path` accordingly.

Run model training with the following command:

```bash
allennlp train ./xbert_tasks/sst2_sentiment_clf.jsonnet -s <MODEL_DIR> --include-package xbert_tasks
```


## Notebook

Change `SST_DATASET_PATH` to the dataset path and `MODEL_DIR` to the directory of the persisted model (e.g. as specified in the above training command).
