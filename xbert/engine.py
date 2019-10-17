from typing import List, Tuple, Callable

from collections import defaultdict
import random
import numpy as np

from xbert import InputInstance, OccludedInstance, Config


def average_relevance_scoring(p_original, p_replaced, n_samples, method):
    #takes a relevance scoring method and applies it to original and samples probabilities
    #output is the difference of original and average of the replaced values
    return method(p_original) - sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples


def std_relevance_scoring(p_original, p_replaced, n_samples, method):
    #takes a relevance scoring method and applies it to samples probabilities
    #output is the std of the replaced values
    average = sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples
    return np.sqrt(sum([(method(probability)-average)**2 * weight for probability, weight in p_replaced]) / n_samples)


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

        occluded_instances = []
        for instance in input_instances:
            occluded_instances += strategy.occluded_instances(instance)

        instance_probabilities = []
        for i in range(0, len(occluded_instances), batch_size):
            batch_candidates = occluded_instances[i: i + batch_size]
            instance_probabilities += self.batcher(batch_candidates)

        return occluded_instances, instance_probabilities

    def relevances(self, occluded_instances, instance_probabilities, std=False, scoring_method=lambda x: x):
        #calculates relevance by average or standard deviation
        #default scoring of the candidates is the difference of prediction

        positional_probabilities = defaultdict(lambda: defaultdict(list))
        for instance, p_instance in zip(occluded_instances, instance_probabilities):
            positional_probabilities[instance.id][instance.occluded_indices].append((p_instance, instance.weight))

        relevances = defaultdict(lambda: defaultdict(float))
        n_samples = getattr(self.config.strategy, "n_samples", 1)

        for input_id, input_probabilities in positional_probabilities.items():
            for position, probabilities_weights_tuple_list in input_probabilities.items():

                # skip relevance computation for original input
                if position is None:
                    continue

                assert len(input_probabilities[None]) == 1

                p_original = input_probabilities[None][0][0]

                if std:
                    relevance = std_relevance_scoring(p_original,
                                                      probabilities_weights_tuple_list,
                                                      n_samples,
                                                      scoring_method)
                else:
                    relevance = average_relevance_scoring(p_original,
                                                          probabilities_weights_tuple_list,
                                                          n_samples,
                                                          scoring_method)

                relevances[input_id][position] = relevance

        return relevances

    # receive input sentences (original) + target(s) as List[Tuple[str, int]]
    # prepare candidates for each input sentence List[Tuple[List[str], int, int, float]] (tokens, input sentence id, replaced index, weight)
    # batch and forward batches to "batcher" method as List[List[str]]
    # receive probabilities for each candidate sentence
    # compute relevance per index as p_c = p(replaced index = -1), p_c_w = sum(replaced index = i) / n_samples
    # and relevance = np.log2(odds(p_c)) - np.log2(odds(p_c_w)) with odds => p / (1 - p)
    # return results as List[List[str], List[float]] (tokens, relevances)