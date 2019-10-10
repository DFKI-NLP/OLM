from typing import List

from xbert import InputInstance, OccludedInstance
from xbert.occlusion import Strategy


@Strategy.register("unk_replacement")
class UnkReplacement(Strategy):
    def __init__(self, unk_token: str):
        self.unk_token = unk_token

    def occluded_instances(self,
                           input_instance: InputInstance) -> List[OccludedInstance]:
        # add original sentence to candidates
        occluded_instances = [OccludedInstance.from_input_instance(input_instance)]

        for field_name, token_field in input_instance.token_fields.items():
            for token_idx in range(len(token_field.tokens)):
                occluded_inst = OccludedInstance.from_input_instance(
                        input_instance,
                        occlude_token=self.unk_token,
                        occlude_field_index=(field_name, token_idx),
                        weight=1.)
                occluded_instances.append(occluded_inst)

        return occluded_instances
