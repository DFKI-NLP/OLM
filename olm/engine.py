from typing import List, Tuple, Callable

import time
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from olm import InputInstance, OccludedInstance, Config


def weight_of_evidence(p):
    #definition taken from http://lkm.fri.uni-lj.si/rmarko/papers/RobnikSikonjaKononenko08-TKDE.pdf
    #and https://arxiv.org/abs/1702.04595
    return np.log2(p / (1. + 1e-12 - p))


def difference_of_log_probabilities(p):
    #shows how much the cross entropy of the true label changes with sampled replacements
    return np.log(p + 1e-12)


def calculate_correlation(relevance_dict_1, relevance_dict2):
    #calculates correlation of relevances of two methods by input and averages these
    assert relevance_dict_1.keys() == relevance_dict2.keys()

    zipped_value_dict = defaultdict(list)
    for sentence_key, sentence_value_list in relevance_dict_1.items():
        for word_key, relevance in sentence_value_list.items():
            zipped_value_dict[sentence_key].append([relevance, relevance_dict2[sentence_key][word_key]])

    result = [0, 0]
    for sentence_zip_array in zipped_value_dict.values():
        sentence_relevance_correlation = np.corrcoef(np.array(sentence_zip_array), rowvar=False)[0][1]
        if not np.isnan(sentence_relevance_correlation):
            result[0] += sentence_relevance_correlation
            result[1] += 1

    return result[0]/result[1]


class Engine:
    def __init__(self,
                 config: Config,
                 batcher: Callable[[List[OccludedInstance]], List[float]],
                 prepare=None) -> None:
        self.config = config
        self.batcher = batcher
        self.prepare = prepare

    def run(self, input_instances: List[InputInstance]) -> List[Tuple[List[str],
                                                                      List[float]]]:
        strategy = self.config.strategy
        batch_size = self.config.batch_size

        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        candidate_instances = []
        # time1 = time.time()
        for instance in tqdm(input_instances):
            candidate_instances += strategy.get_candidate_instances(instance)
        # time2 = time.time()
        # print('{:s} took {:.3f} ms'.format("CANDIDATES", (time2-time1)*1000.0))
        # print("Num candidate instances: ", len(candidate_instances))

        candidate_results = []
        for i in tqdm(range(0, len(candidate_instances), batch_size), total=len(candidate_instances) // batch_size):
            batch_candidates = candidate_instances[i: i + batch_size]
            # time1 = time.time()
            candidate_results += self.batcher(batch_candidates)
            # time2 = time.time()
            # print('{:s} took {:.3f} ms'.format("INFERENCE", (time2-time1)*1000.0))

        return candidate_instances, candidate_results

    def relevances(self, candidate_instances, candidate_results):
        return self.config.strategy.relevances(candidate_instances, candidate_results)
