from xbert import InputInstance
from xbert.occlusion.strategies import UnkReplacement

UNK_TOKEN = "__UNK__"


def test_unk_replacement():
    input_instance = InputInstance(id_=1,
                                   sent1=["a", "b", "c"],
                                   sent2=["d", "e", "f"])
    strategy = UnkReplacement(unk_token=UNK_TOKEN)

    candidate_instances = strategy.get_candidate_instances(input_instance)

    assert len(candidate_instances) == 7

    instance = candidate_instances[1]
    assert instance.sent1.tokens == [UNK_TOKEN, "b", "c"]
    assert instance.sent2.tokens == input_instance.sent2.tokens
