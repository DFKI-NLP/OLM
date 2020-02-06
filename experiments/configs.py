# RoBERTa config for all tasks
ROBERTA_UNK_CONFIG = {
        "strategy": "unk_replacement",
        "batch_size": 128,
        "unk_token": "<unk>",
        "seed": 1111,
}

ROBERTA_DEL_CONFIG = {
        "strategy": "delete",
        "batch_size": 128,
        "seed": 1111,
}

ROBERTA_RESAMPLING_CONFIG = {
        "strategy": "bert_lm_sampling",
        "std": False,
        "cuda_device": 0,
        "bert_model": "bert-base-uncased",
        "batch_size": 128,
        "n_samples": 100,
        "verbose": False,
        "seed": 1111,
}

ROBERTA_RESAMPLING_STD_CONFIG = {
        "strategy": "bert_lm_sampling",
        "std": True,
        "cuda_device": 0,
        "bert_model": "bert-base-uncased",
        "batch_size": 128,
        "n_samples": 100,
        "verbose": False,
        "seed": 1111,
}

ROBERTA_GRADIENT_CONFIG = {
        "strategy": "gradient",
        "batch_size": 2,
        "seed": 1111,
}

# SST2 for AllenNLP
SST2_UNK_CONFIG = {
        "strategy": "unk_replacement",
        "batch_size": 128,
        "unk_token": "__unk__",
        "seed": 1111,
}

SST2_DEL_CONFIG = {
        "strategy": "delete",
        "batch_size": 128,
        "seed": 1111,
}

SST2_RESAMPLING_CONFIG = {
        "strategy": "bert_lm_sampling",
        "std": False,
        "cuda_device": 0,
        "bert_model": "bert-base-uncased",
        "batch_size": 128,
        "n_samples": 100,
        "verbose": False,
        "seed": 1111,
}

SST2_RESAMPLING_STD_CONFIG = {
        "strategy": "bert_lm_sampling",
        "std": True,
        "cuda_device": 0,
        "bert_model": "bert-base-uncased",
        "batch_size": 128,
        "n_samples": 100,
        "verbose": False,
        "seed": 1111,
}

SST2_GRADIENT_CONFIG = {
        "strategy": "gradient",
        "batch_size": 2,
        "seed": 1111,
}
