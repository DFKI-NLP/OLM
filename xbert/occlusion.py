from typing import List, Tuple, Dict, Any

import numpy as np
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer

from xbert.modeling import BertForMaskedLMLayer
from xbert.candidates import get_candidates


def average_relevance_scoring(p_original, p_replaced, n_samples, method):
    #takes a relevance scoring method and applies it to original and samples probabilities
    #output is the difference of original and average of the replaced values
    return method(p_original) - sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples

def variance_relevance_scoring(p_original, p_replaced, n_samples, method):
    #takes a relevance scoring method and applies it to samples probabilities
    #output is the variance of the replaced values
    average = sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples
    return sum([(method(probability)-average)**2 * weight for probability, weight in p_replaced]) / n_samples

def weight_of_evidence(p):
    #definition taken from http://lkm.fri.uni-lj.si/rmarko/papers/RobnikSikonjaKononenko08-TKDE.pdf
    #and https://arxiv.org/abs/1702.04595
    return np.log2(p / (1. + 1e-12 - p))

def difference_of_log_probabilities(p):
    #shows how much the cross entropy of the true label changes with sampled replacements
    return np.log(p + 1e-12)


class Engine:
    def __init__(self, params: Dict[str, Any], batcher, prepare = None) -> None:
        self.params = params
        self.batcher = batcher
        self.prepare = prepare

        bert_model = self.params.get("bert_model", "bert-base-uncased")
        cuda_device = self.params.get("cuda_device", -1)

        bert = BertForMaskedLMLayer.from_pretrained(bert_model)
        bert.eval()
        self.bert = bert.to(cuda_device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def run(self, inputs: List[Tuple[int, List[str]]]) -> List[Tuple[List[str], List[float]]]:
        verbose = self.params.get("verbose", False)

        cuda_device = self.params.get("cuda_device", -1)
        batch_size = self.params.get("batch_size", 32)
        n_samples = self.params.get("n_samples")
        unknown = self.params.get("unknown")

        candidates = []
        for input_id, tokens in inputs:
            candidates += get_candidates(tokens=tokens,
                                         input_id=input_id,
                                         bert=self.bert,
                                         tokenizer=self.tokenizer,
                                         n_samples=n_samples,
                                         replace_subwords=False,
                                         unknown=unknown,
                                         cuda_device=cuda_device,
                                         verbose=verbose)

        candidate_probabilities = []
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i: i + batch_size]
            candidate_probabilities += self.batcher(batch_candidates)

        self.positional_probabilities = defaultdict(lambda: defaultdict(list))
        for candidate, p_candidate in zip(candidates, candidate_probabilities):
            self.positional_probabilities[candidate.id][candidate.replaced_index].append((p_candidate, candidate.weight))
        
        return candidates

    def relevances(self, variance=False, scoring_method=lambda x:x):
        #calculates relevance by average or variance
        #default scoring of the candidates is the difference of prediction
        relevances = defaultdict(lambda: defaultdict(float))
        n_samples = self.params.get("n_samples")

        for input_id, input_probabilities in self.positional_probabilities.items():
            for position, probabilities_weights_tuple_list in input_probabilities.items():

                # skip relevance computation for original input
                if position == -1:
                    continue

                assert len(input_probabilities[-1]) == 1

                p_original = input_probabilities[-1][0][0]

                if variance:
                    relevance = variance_relevance_scoring(p_original,
                        probabilities_weights_tuple_list, n_samples, scoring_method)
                else:
                    relevance = average_relevance_scoring(p_original,
                        probabilities_weights_tuple_list, n_samples, scoring_method)

                relevances[input_id][position] = relevance

        return relevances

    # receive input sentences (original) + target(s) as List[Tuple[str, int]]
    # prepare candidates for each input sentence List[Tuple[List[str], int, int, float]] (tokens, input sentence id, replaced index, weight)
    # batch and forward batches to "batcher" method as List[List[str]]
    # receive probabilities for each candidate sentence
    # compute relevance per index as p_c = p(replaced index = -1), p_c_w = sum(replaced index = i) / n_samples
    # and relevance = np.log2(odds(p_c)) - np.log2(odds(p_c_w)) with odds => p / (1 - p)
    # return results as List[List[str], List[float]] (tokens, relevances)