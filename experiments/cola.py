from typing import List, Dict, Tuple

import os
import argparse
import dill
import json
from functools import partial
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from segtok.tokenizer import web_tokenizer

from xbert import InputInstance, Config
from xbert.engine import Engine
from xbert.occlusion.explainer import (VanillaGradExplainer, GradxInputExplainer,
                                       SaliencyExplainer, IntegrateGradExplainer)
from configs import (ROBERTA_UNK_CONFIG, ROBERTA_RESAMPLING_CONFIG,
                     ROBERTA_RESAMPLING_STD_CONFIG, ROBERTA_GRADIENT_CONFIG,
                     ROBERTA_DEL_CONFIG)
from utils import collate_tokens


OCCLUSION_STRATEGIES = ["unk", "delete", "resampling", "resampling_std"]
GRAD_STRATEGIES = ["grad", "gradxinput", "saliency", "integratedgrad"]
ALL_STRATEGIES = OCCLUSION_STRATEGIES + GRAD_STRATEGIES


def byte_pair_offsets(input_ids, tokenizer):
    def get_offsets(tokens, start_offset):
        offsets = [start_offset]
        for t_idx, token in enumerate(tokens, start_offset):
            if not token.startswith(" "):
                continue
            offsets.append(t_idx)
        offsets.append(start_offset + len(tokens))
        return offsets

    tokens = [tokenizer.convert_tokens_to_string(t)
              for t in tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)]
    tokens = [token for token in tokens if token != "<pad>"]
    tokens = tokens[1:-1]

    offsets = get_offsets(tokens, start_offset=1)

    return offsets


def read_cola_dataset(path: str) -> List[Tuple[List[str], str]]:
    dataset = []
    with open(path) as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent, target = tokens[3], tokens[1]
            dataset.append((sent, target))

    return dataset


def dataset_to_input_instances(dataset: List[Tuple[List[str], str]]) -> List[InputInstance]:
    input_instances = []
    for idx, (sent, _) in enumerate(dataset):
        instance = InputInstance(id_=idx, sent=web_tokenizer(sent))
        input_instances.append(instance)

    return input_instances


def get_labels(dataset: List[Tuple[List[str], List[str], str]]) -> List[str]:
    return [int(label) for _, label in dataset]


def encode_instance(input_instance, tokenizer):
    return tokenizer.encode(text=" ".join(input_instance.sent.tokens),
                            add_special_tokens=True,
                            return_tensors="pt")[0]


def predict(input_instances, model, tokenizer, cuda_device):
    if isinstance(input_instances, InputInstance):
        input_instances = [input_instances]

    input_ids = [encode_instance(instance, tokenizer) for instance in input_instances]
    attention_mask = [torch.ones_like(t) for t in input_ids]

    input_ids = collate_tokens(input_ids, pad_idx=1).to(cuda_device)
    attention_mask = collate_tokens(attention_mask, pad_idx=0).to(cuda_device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    return F.softmax(logits, dim=-1)


def batcher_occlusion(batch_instances, labels, tokenizer, model, cuda_device):
    true_label_indices = []
    probabilities = []
    with torch.no_grad():
        probs = predict(batch_instances, model, tokenizer, cuda_device).cpu().numpy().tolist()
        for batch_idx, instance in enumerate(batch_instances):
            # the instance id is also the position in the list of labels
            idx = instance.id
            true_label_idx = labels[idx]
            true_label_indices.append(true_label_idx)
            probabilities.append(probs[batch_idx][true_label_idx])

    return probabilities


def batcher_gradient(batch_instances, labels, tokenizer, model, explainer, cuda_device):
    input_ids = [encode_instance(instance, tokenizer) for instance in batch_instances]
    attention_mask = [torch.ones_like(t) for t in input_ids]

    input_ids = collate_tokens(input_ids, pad_idx=1).to(cuda_device)
    attention_mask = collate_tokens(attention_mask, pad_idx=0).to(cuda_device)

    inputs_embeds = model.roberta.embeddings(input_ids=input_ids).detach()

    true_label_idx_list = [labels[instance.id] for instance in batch_instances]
    true_label_idx_tensor = torch.tensor(true_label_idx_list, dtype=torch.long, device=cuda_device)

    inputs_embeds.requires_grad = True
    expl = explainer.explain(inp={"inputs_embeds": inputs_embeds, "attention_mask": attention_mask},
                             ind=true_label_idx_tensor)

    input_ids_np = input_ids.cpu().numpy()
    expl_np = expl.cpu().numpy()

    relevances = []
    for b_idx in range(input_ids_np.shape[0]):
        offsets = byte_pair_offsets(input_ids_np[b_idx].tolist(), tokenizer)

        relevance_dict = defaultdict(float)
        for token_idx, (token_start, token_end) in enumerate(zip(offsets, offsets[1:])):
            relevance = expl_np[b_idx][token_start: token_end].sum()
            relevance_dict[("sent", token_idx)] = relevance
        relevances.append(relevance_dict)

    return relevances


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files for the CoLA task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--strategy", default=None, type=str, required=True,
                        help="The explainability strategy to use.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the results will be written.")

    # Other parameters
    parser.add_argument("--do_run", action='store_true',
                        help="Whether to run the explainability strategy.")
    parser.add_argument("--do_relevances", action='store_true',
                        help="Whether to compute relevances from the run results.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="The cache dir. Should contain the candidate_instances.pkl file of a strategy.")

    # Optional parameters
    parser.add_argument("--cuda_device", default=0, type=int,
                        help="The default cuda device.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")

    args = parser.parse_args()

    if args.strategy.lower() not in ALL_STRATEGIES:
        raise ValueError("Explainability strategy: '{}' not in {}".format(args.strategy, ALL_STRATEGIES))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. "
    #                      "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = read_cola_dataset(os.path.join(args.data_dir, "dev.tsv"))
    input_instances = dataset_to_input_instances(dataset)
    labels = get_labels(dataset)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.cuda_device)

    if args.strategy.lower() in GRAD_STRATEGIES:
        config_dict = ROBERTA_GRADIENT_CONFIG
        config = Config.from_dict(config_dict)

        # output_getter extracts the first entry of the return tuple and also applies a softmax to the
        # log probabilities
        explainer = {
                "grad": VanillaGradExplainer,
                "gradxinput": GradxInputExplainer,
                "saliency": SaliencyExplainer,
                "integratedgrad": IntegrateGradExplainer,
        }[args.strategy](model=model,
                         input_key="inputs_embeds",
                         output_getter=lambda x: F.softmax(x[0], dim=-1))

        batcher = partial(batcher_gradient,
                          labels=labels,
                          tokenizer=tokenizer,
                          model=model,
                          explainer=explainer,
                          cuda_device=args.cuda_device)
    else:
        config_dict = {
                "unk": ROBERTA_UNK_CONFIG,
                "delete": ROBERTA_DEL_CONFIG,
                "resampling": ROBERTA_RESAMPLING_CONFIG,
                "resampling_std": ROBERTA_RESAMPLING_STD_CONFIG,
        }[args.strategy.lower()]
        config = Config.from_dict(config_dict)

        batcher = partial(batcher_occlusion,
                          labels=labels,
                          tokenizer=tokenizer,
                          model=model,
                          cuda_device=args.cuda_device)

    engine = Engine(config, batcher)

    candidate_results_file = os.path.join(args.output_dir, "candidate_instances.pkl")

    with open(os.path.join(args.output_dir, "args.json"), "w") as out_f:
        json.dump(vars(args), out_f)

    with open(os.path.join(args.output_dir, "config.json"), "w") as out_f:
        json.dump(config_dict, out_f)

    if args.do_run:
        candidate_instances, candidate_results = engine.run(input_instances)
        with open(candidate_results_file, "wb") as out_f:
            dill.dump((candidate_instances, candidate_results), out_f)

    if args.do_relevances:
        if args.cache_dir is not None:
            candidate_results_file = os.path.join(args.cache_dir, "candidate_instances.pkl")

        with open(candidate_results_file, "rb") as in_f:
            candidate_instances, candidate_results = dill.load(in_f)

        relevances = engine.relevances(candidate_instances, candidate_results)

        with open(os.path.join(args.output_dir, "relevances.pkl"), "wb") as out_f:
            dill.dump(relevances, out_f)


if __name__ == "__main__":
    main()
