import numpy as np

from step4_model_training.paper_metrics import bio_to_io_probabilities, classification_measures


def test_measures_match_hand_computation():
    # classes: 0 = O (negative), 1 = DISEASE, 2 = DRUG
    gold = [0, 1, 1, 2, 0, 2]
    pred = [0, 1, 0, 2, 1, 2]
    probs = np.array([[.9, .05, .05], [.1, .8, .1], [.6, .3, .1], [.1, .1, .8], [.3, .6, .1], [.2, .1, .7]])
    m = classification_measures(gold, pred, probs, ["O", "DISEASE", "DRUG"], negative_class=0)
    disease = m["per_class"]["DISEASE"]
    assert (disease["tp"], disease["fp"], disease["fn"], disease["tn"]) == (1, 1, 1, 3)
    assert disease["specificity"] == 0.75
    assert m["per_class"]["DRUG"]["f1"] == 1.0
    assert abs(m["accuracy"] - 4 / 6) < 1e-9
    assert m["auc"] is not None and 0.5 < m["auc"] <= 1.0


def test_bio_probabilities_are_merged():
    probs = np.array([[0.1, 0.5, 0.4]])
    merged = bio_to_io_probabilities(probs, ["O", "B-DRUG", "I-DRUG"], ["O", "DRUG"])
    assert np.allclose(merged, [[0.1, 0.9]])
