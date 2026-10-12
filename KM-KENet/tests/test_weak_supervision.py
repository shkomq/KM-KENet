import numpy as np
import pytest

from common.config import load_config
from common.lexicon import load_lexicon
from common.schema import ENTITY_TYPES, OUTSIDE, bio_to_spans, candidate_pairs, spans_to_bio
from step2_preprocessing.text_tools import build_text_tools, load_stopwords
from step3_weak_supervision.label_matrix import ABSTAIN, NoisyLabelGenerator
from step3_weak_supervision.label_model import DawidSkeneLabelModel, MajorityVoteLabelModel, apply_anchor_votes
from step3_weak_supervision.labeling_functions import (
    LexiconMatcher,
    build_ner_labeling_functions,
    build_relation_lfs,
)
from step3_weak_supervision.labeling_functions.heuristic import DosageSpanLF
from step3_weak_supervision.refinement import NERLabelRefiner, RelationLabeler


FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "kmc_sample.json"


@pytest.fixture(scope="module")
def resources():
    cfg = load_config()
    tools = build_text_tools(cfg)
    lexicon = load_lexicon(cfg.path("lexicon"))
    stopwords = load_stopwords(cfg.path("stopwords"), tools)
    return tools, LexiconMatcher(lexicon, tools), stopwords


def test_schema_matches_paper_entity_categories():
    assert set(ENTITY_TYPES) == {"DISEASE", "SYMPTOM", "DRUG", "DOSAGE", "PROCEDURE", "BODY_PART"}
    tags = spans_to_bio([(0, 2, "DRUG"), (3, 5, "DOSAGE")], 6)
    assert bio_to_spans(tags) == [(0, 2, "DRUG"), (3, 5, "DOSAGE")]
    assert (0, 1, "HAS_DOSAGE") in candidate_pairs(["DRUG", "DOSAGE"])


def test_lexicon_matches_inflected_forms(resources):
    tools, matcher, _ = resources
    tokens = tools.tokenize("نەخۆشی شەکرە کاریگەری لەسەر گورچیلەکان هەیە")
    found = {(m.entry.entry_id, m.entry.entity_type) for m in matcher.find(tokens)}
    assert ("diabetes", "DISEASE") in found and ("kidney", "BODY_PART") in found


def test_dosage_span_lf():
    votes = DosageSpanLF()(["پاراسیتامۆڵ", "500", "mg", "وەربگرە"])
    assert votes == [None, "DOSAGE", "DOSAGE", None]


def test_end_to_end_weak_labels(resources):
    """Label model fitted on the synthetic corpus, then the three probe sentences are checked."""
    from step1_data_collection.collectors import load_commented_json

    tools, matcher, stopwords = resources
    sentences = [
        "ئەمۆکسیسیلین بۆ چارەسەری سیل بەکاردێت .",
        "نیشانەکانی ڕەبۆ بریتین لە کۆکە و تەنگەنەفەسی .",
        "نەخۆش دەتوانێت پاراسیتامۆڵ 500 mg دوو جار وەربگرێت .",
    ]
    corpus = [
        tools.tokenizer.tokenize(sentence)
        for record in load_commented_json(FIXTURE)
        for sentence in tools.tokenizer.split_sentences(tools.normalizer.normalize(record["response"]))
    ]
    tokenized = [tools.tokenize(s) for s in sentences]
    classes = [OUTSIDE, *ENTITY_TYPES]
    lfs = build_ner_labeling_functions(matcher, tools.stemmer, stopwords)
    generator = NoisyLabelGenerator(lfs, classes)
    model = DawidSkeneLabelModel(len(classes)).fit(generator.generate(corpus, show_progress=False).matrix)
    matrix = generator.generate(tokenized, show_progress=False)
    labels, confidences = model.predict(matrix.matrix)
    apply_anchor_votes(matrix.matrix, labels, confidences, [matrix.lf_names.index("heuristic_dosage_span")])
    records = [{"sentence_id": f"s{i}", "doc_id": "d", "group": "g", "text": s, "tokens": t}
               for i, (s, t) in enumerate(zip(sentences, tokenized))]
    refiner = NERLabelRefiner(classes, stopwords, tools.stemmer)
    annotations = refiner.build_annotations(records, labels, confidences, matrix.covered, matrix.offsets)
    labeler = RelationLabeler(build_relation_lfs(25))
    for annotation in annotations:
        annotation.relations = labeler.label(annotation)

    types = [{e.type for e in a.entities} for a in annotations]
    assert {"DRUG", "DISEASE"} <= types[0]
    assert {"DISEASE", "SYMPTOM"} <= types[1]
    assert {"DRUG", "DOSAGE"} <= types[2]
    relations = [{r["type"] for r in a.relations} for a in annotations]
    assert "TREATS" in relations[0]
    assert "HAS_SYMPTOM" in relations[1]
    assert "HAS_DOSAGE" in relations[2]


def test_label_models_agree_on_unanimous_votes():
    matrix = np.array([[1, 1, ABSTAIN], [2, ABSTAIN, 2], [ABSTAIN, ABSTAIN, ABSTAIN]], dtype=np.int8)
    for model in (MajorityVoteLabelModel(3), DawidSkeneLabelModel(3, max_iter=10, model_abstains=False)):
        labels, _ = model.fit(matrix).predict(matrix)
        assert labels.tolist()[:2] == [1, 2]
