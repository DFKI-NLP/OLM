{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
  "model": {
    "type": "text_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "text_encoder": {
      "type": "cnn",
      "embedding_dim": 300,
      "num_filters": 150,
      "ngram_filter_sizes": [2, 3, 4],
    },
    // "text_encoder": {
    //   "type": "lstm",
    //   "input_size": 300,
    //   "hidden_size": 150,
    //   "num_layers": 2,
    //   "bidirectional": true
    // },

    "classifier_feedforward": {
      // "input_dim": 300,
      "input_dim": 450,
      "num_layers": 2,
      "hidden_dims": [200, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}