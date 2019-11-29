MNLI_ROBERTA_UNK_CONFIG = {
        "strategy": "unk_replacement",
        "batch_size": 128,
        "unk_token": "<unk>",
        "seed": 1111,
}

MNLI_ROBERTA_RESAMPLING_CONFIG = {
        "strategy": "bert_lm_sampling",
        "cuda_device": 0,
        "bert_model": "bert-base-uncased",
        "batch_size": 128,
        "n_samples": 100,
        "verbose": False,
        "seed": 1111,
}

MNLI_ROBERTA_GRADIENT_CONFIG = {
        "strategy": "gradient",
        "batch_size": 128,
        "seed": 1111,
}
