from typing import List

from overrides import overrides

from olm import InputInstance, OccludedInstance
from olm.occlusion import Strategy, OcclusionStrategy


@Strategy.register("delete")
class Delete(OcclusionStrategy):
    def __init__(self,
                 std: bool = False,
                 scoring_method=lambda x: x):
        super().__init__(n_samples=1, std=std, scoring_method=scoring_method)

    @overrides
    def get_candidate_instances(self,
                                input_instance: InputInstance) -> List[OccludedInstance]:
        # add original sentence to candidates
        occluded_instances = [OccludedInstance.from_input_instance(input_instance)]

        for field_name, token_field in input_instance.token_fields.items():
            for token_idx in range(len(token_field.tokens)):
                occluded_inst = OccludedInstance.from_input_instance(
                        input_instance,
                        occlude_field_index=(field_name, token_idx),
                        weight=1.)
                occluded_instances.append(occluded_inst)

        return occluded_instances
