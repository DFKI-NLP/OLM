from typing import List, Dict, Tuple

import os
import argparse
import dill
import json
from functools import partial
from collections import defaultdict

import torch
import torch.nn.functional as F

from allennlp.models.archival import load_archive
from allennlp.data import Instance
from allennlp.predictors import Predictor

from xbert import InputInstance, Config
from xbert.engine import Engine
from xbert.occlusion.explainer_allennlp import (AllenNLPVanillaGradExplainer,
                                                AllenNLPGradxInputExplainer,
                                                AllenNLPSaliencyExplainer,
                                                AllenNLPIntegrateGradExplainer)
from configs import (SST2_UNK_CONFIG, SST2_RESAMPLING_CONFIG,
                     SST2_RESAMPLING_STD_CONFIG, SST2_GRADIENT_CONFIG,
                     SST2_DEL_CONFIG)


OCCLUSION_STRATEGIES = ["unk", "delete", "resampling", "resampling_std"]
GRAD_STRATEGIES = ["grad", "gradxinput", "saliency", "integratedgrad"]
ALL_STRATEGIES = OCCLUSION_STRATEGIES + GRAD_STRATEGIES


def dataset_to_input_instances(dataset: List[Instance]) -> List[InputInstance]:
    return [InputInstance(id_=idx, text=[t.text for t in instance.fields["tokens"].tokens])
            for idx, instance in enumerate(dataset)]


def get_labels(dataset: List[Tuple[List[str], List[str], str]]) -> List[str]:
    return [instance.fields["label"].label for instance in dataset]


def batcher_occlusion(batch_instances, labels, predictor):
    label2idx = predictor._model.vocab.get_token_to_index_vocabulary("labels")

    true_label_indices = []
    batch_dicts = []
    for instance in batch_instances:
        idx = instance.id
        true_label_idx = label2idx[labels[idx]]
        true_label_indices.append(true_label_idx)
        batch_dicts.append(dict(text=instance.text.tokens))

    results = predictor.predict_batch_json(batch_dicts)

    return [result["class_probabilities"][tl_idx] for (result, tl_idx)
            in zip(results, true_label_indices)]


def batcher_gradient(batch_instances, labels, predictor, explainer, cuda_device):
    label2idx = predictor._model.vocab.get_token_to_index_vocabulary("labels")

    relevances = []
    for instance in batch_instances:
        idx = instance.id
        true_label_idx = label2idx[labels[idx]]

        inst = predictor._json_to_instance(dict(text=instance.text.tokens))
        expl = explainer.explain([inst], ind=torch.tensor([true_label_idx], dtype=torch.long, device=cuda_device))[0]
        expl_np = expl.sum(dim=-1).squeeze().detach().cpu().numpy().tolist()

        relevance_dict = defaultdict(float)
        for token_idx, relevance in enumerate(expl_np):
            relevance_dict[("text", token_idx)] = relevance
        relevances.append(relevance_dict)

    return relevances


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files for the MNLI task.")
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
    parser.add_argument("--predictor_name", default="sst_text_classifier", type=str,
                        help="The predictor name. Defaults to sst_text_classifier.")

    args = parser.parse_args()

    if args.strategy.lower() not in ALL_STRATEGIES:
        raise ValueError("Explainability strategy: '{}' not in {}".format(args.strategy, ALL_STRATEGIES))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. "
    #                      "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # disable cudnn when running on GPU, because can't do a backward pass when not in train mode
    if args.cuda_device >= 0:
        torch.backends.cudnn.enabled = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    archive = load_archive(os.path.join(args.model_name_or_path, "model.tar.gz"))
    Predictor.from_archive(archive, args.predictor_name)

    dataset = predictor._dataset_reader.read(os.path.join(args.data_dir, "dev.tsv"))

    input_instances = dataset_to_input_instances(dataset)
    labels = get_labels(dataset)

    if args.strategy.lower() in GRAD_STRATEGIES:
        config_dict = SST2_GRADIENT_CONFIG
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
                          predictor=predictor,
                          explainer=explainer,
                          cuda_device=args.cuda_device)
    else:
        config_dict = {
                "unk": SST2_UNK_CONFIG,
                "delete": SST2_DEL_CONFIG,
                "resampling": SST2_RESAMPLING_CONFIG,
                "resampling_std": SST2_RESAMPLING_STD_CONFIG,
        }[args.strategy.lower()]
        config = Config.from_dict(config_dict)

        batcher = partial(batcher_occlusion,
                          labels=labels,
                          predictor=predictor)

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
