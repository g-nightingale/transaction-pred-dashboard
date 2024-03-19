from db.datagenerator import FraudDataGenerator
import numpy as np

def test_datagen_probs():
    fdg = FraudDataGenerator()
    for feature in fdg.features_prob_dict.keys():
        assert np.isclose(sum([v for v in fdg.features_prob_dict[feature].values()]), 1.0)

    fdg.reset_fraud_probabilities()
    assert fdg.features_fraud_rate_dict.keys() == fdg.features_prob_dict.keys()

    fdg.reset_fraud_probabilities()
    assert fdg.features_fraud_rate_dict.keys() == fdg.features_prob_dict.keys()

def test_datagen_num_features():
    fdg = FraudDataGenerator()
    for feature in fdg.features_fraud_rate_dict.keys():
        assert not any(value is None for value in fdg.features_fraud_rate_dict[feature].values())

def test_datagen_data_length():
    RECORDS_TO_GENERATE = 1001
    fdg = FraudDataGenerator()
    new_data = fdg.generate_data(RECORDS_TO_GENERATE)

    assert len(new_data) == RECORDS_TO_GENERATE
