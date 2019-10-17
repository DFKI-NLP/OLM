from typing import List


from xbert import InputInstance, OccludedInstance

STRATEGY_REGISTRY = {}


class Strategy:

    def occluded_instances(self,
                           input_instance: InputInstance) -> List[OccludedInstance]:
        raise NotImplementedError("Strategy must implement 'occluded_instances'.")

    @staticmethod
    def register(strategy_name: str):
        def inner(clazz):
            STRATEGY_REGISTRY[strategy_name] = clazz
            return clazz

        return inner
