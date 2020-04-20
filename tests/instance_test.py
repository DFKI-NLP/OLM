import pytest

from olm import InputInstance, OccludedInstance
from olm.instances import TokenField, OccludedTokenField

SENT_1 = ["a", "b", "c"]
SENT_2 = ["d", "e", "f"]
OCCLUDED_SENT_1 = ["a", "occluded", "c"]


def test_input_instance():
    instance = InputInstance(id_=1, sent1=SENT_1, sent2=SENT_2)

    assert instance.id == 1
    assert instance.sent1.tokens == SENT_1
    assert instance.token_fields["sent1"].tokens == SENT_1
    assert instance.token_fields["sent2"].tokens == SENT_2
    assert instance.sent2.tokens == SENT_2
    assert isinstance(instance.sent1, TokenField)
    assert isinstance(instance.sent2, TokenField)

    occluded_inst = OccludedInstance.from_input_instance(
            instance,
            occlude_token="occluded",
            occlude_field_index=("sent1", 1),
            weight=5)

    assert occluded_inst.id == 1    
    assert occluded_inst.sent1.tokens == OCCLUDED_SENT_1
    assert occluded_inst.token_fields["sent1"].tokens == OCCLUDED_SENT_1
    assert occluded_inst.token_fields["sent2"].tokens == SENT_2
    assert occluded_inst.sent2.tokens == SENT_2
    assert isinstance(occluded_inst.sent1, OccludedTokenField)
    assert isinstance(occluded_inst.sent2, TokenField)


def test_occluded_instance():
    instance = InputInstance(id_=1, sent1=SENT_1, sent2=SENT_2)

    with pytest.raises(ValueError) as excinfo:
        OccludedInstance.from_input_instance(
                instance,
                occlude_token="occluded",
                weight=5)
    assert "'occlude_token' requires setting 'occlude_field_index'" in str(excinfo.value)

    occluded_inst = OccludedInstance.from_input_instance(
            instance,
            occlude_token="occluded",
            occlude_field_index=("sent1", 1),
            weight=5)

    assert occluded_inst.sent1.occluded_index == 1

    assert occluded_inst.occluded_indices == ("sent1", 1)
