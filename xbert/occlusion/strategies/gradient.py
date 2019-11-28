from typing import List, Dict

from xbert import InputInstance, OccludedInstance
from xbert.occlusion import Strategy, GradientStrategy


@Strategy.register("gradient")
class Gradient(GradientStrategy):
    def __init__(self):
        super().__init__()

    def relevances(self, candidate_instances, candidate_results) -> Dict[int, Dict[str, float]]:
        return {instance.id: result for instance, result in zip(candidate_instances, candidate_results)}
