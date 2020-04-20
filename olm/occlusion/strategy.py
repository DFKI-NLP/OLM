from typing import List, Dict

from collections import defaultdict
from overrides import overrides
import numpy as np

from olm import InputInstance, OccludedInstance

STRATEGY_REGISTRY = {}


class Strategy:
    def get_candidate_instances(self,
                                input_instance: InputInstance) -> List[OccludedInstance]:
        raise NotImplementedError("Strategy must implement 'get_candidate_instances'.")

    def relevances(self, candidate_instances, candidate_results) -> Dict[int, Dict[str, float]]:
        raise NotImplementedError("Strategy must implement 'relevances'.")

    @staticmethod
    def register(strategy_name: str):
        def inner(clazz):
            STRATEGY_REGISTRY[strategy_name] = clazz
            return clazz

        return inner


def average_relevance_scoring(p_original, p_replaced, n_samples, method):
    #takes a relevance scoring method and applies it to original and samples probabilities
    #output is the difference of original and average of the replaced values
    return method(p_original) - sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples


def std_relevance_scoring(p_original, p_replaced, n_samples, method):
    #takes a relevance scoring method and applies it to samples probabilities
    #output is the std of the replaced values
    average = sum([method(probability)*weight for probability, weight in p_replaced]) / n_samples
    return np.sqrt(sum([(method(probability)-average)**2 * weight for probability, weight in p_replaced]) / n_samples)


class OcclusionStrategy(Strategy):
    def __init__(self, n_samples, std: bool, scoring_method):
        #calculates relevance by average or standard deviation
        #default scoring of the candidates is the difference of prediction
        super().__init__()
        self.n_samples = n_samples
        self.std = std
        self.scoring_method = scoring_method

    @overrides
    def relevances(self, candidate_instances, candidate_results) -> Dict[int, Dict[str, float]]:
        positional_probabilities = defaultdict(lambda: defaultdict(list))
        for instance, p_instance in zip(candidate_instances, candidate_results):
            positional_probabilities[instance.id][instance.occluded_indices].append((p_instance, instance.weight))

        relevances = defaultdict(lambda: defaultdict(float))

        for input_id, input_probabilities in positional_probabilities.items():
            for position, probabilities_weights_tuple_list in input_probabilities.items():

                # skip relevance computation for original input
                if position is None:
                    continue

                assert len(input_probabilities[None]) == 1

                p_original = input_probabilities[None][0][0]

                if self.std:
                    relevance = std_relevance_scoring(p_original,
                                                      probabilities_weights_tuple_list,
                                                      self.n_samples,
                                                      self.scoring_method)
                else:
                    relevance = average_relevance_scoring(p_original,
                                                          probabilities_weights_tuple_list,
                                                          self.n_samples,
                                                          self.scoring_method)

                relevances[input_id][position] = relevance
        return relevances


class GradientStrategy(Strategy):
    @overrides
    def get_candidate_instances(self,
                                input_instance: InputInstance) -> List[OccludedInstance]:
        # for gradient based strategies, we only need the original sentence
        return [OccludedInstance.from_input_instance(input_instance)]
