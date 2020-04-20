from olm import Config, InputInstance, Engine

TEST_CONFIG = {
        "strategy": "unk_replacement",
        "unk_token": "__UNK__",
        "seed": 1234
}


def batcher(batch_instances):
    return [0.1 * i for i in range(len(batch_instances))]


def test_engine():
    config = Config.from_dict(TEST_CONFIG)
    engine = Engine(config, batcher=batcher)

    input_instance = InputInstance(id_=1,
                                   sent1=["a", "b", "c"],
                                   sent2=["d", "e", "f"])

    occluded_instances, instance_probabilities = engine.run([input_instance])

    assert len(occluded_instances) == 7

    relevances = engine.relevances(occluded_instances, instance_probabilities)
