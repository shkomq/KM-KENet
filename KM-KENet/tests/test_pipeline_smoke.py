"""End-to-end smoke test: steps 1-6, evaluation and cross-validation on the synthetic sample.

Uses a tiny random XLM-R (built offline by tests/make_tiny_encoder.py), no Aya and
no dense retriever, so it runs on a CPU in about a minute without internet access.
It checks that every step runs and writes its outputs — not the model's accuracy.

    pytest tests/test_pipeline_smoke.py -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES = TESTS_DIR / "fixtures"


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("kmkenet")


@pytest.fixture(scope="module")
def overrides(workdir: Path) -> list[str]:
    from tests.make_tiny_encoder import build

    encoder = build(FIXTURES / "kmc_sample.json", workdir / "tiny-xlmr")
    values = {
        "paths.corpus_root": str(FIXTURES),
        "paths.interim_dir": str(workdir / "data/interim"),
        "paths.processed_dir": str(workdir / "data/processed"),
        "paths.weak_dir": str(workdir / "data/weak_labels"),
        "paths.model_dir": str(workdir / "models/km_kenet"),
        "paths.kg_dir": str(workdir / "outputs/knowledge_graph"),
        "paths.answers_dir": str(workdir / "outputs/answers"),
        "paths.cache_dir": str(workdir / "outputs/cache"),
        "paths.logs_dir": str(workdir / "logs"),
        "data_collection.kmc_json.files": "[kmc_sample.json]",
        "model.encoder": str(encoder),
        "model.max_length": "64",
        "model.relation_hidden_size": "32",
        "training.epochs": "2",
        "training.batch_size": "16",
        "training.fp16": "false",
        "training.max_train_examples": "null",
        "project.device": "cpu",
        "knowledge_graph.min_triple_support": "1",
        "knowledge_graph.exports.visualization": "false",
        "retrieval.use_dense_retrieval": "false",
        "retrieval.aya.backend": "none",
        "evaluation.cross_validation.folds": "2",
        "evaluation.cross_validation.epochs": "1",
        "evaluation.cross_validation.max_examples": "300",
    }
    args: list[str] = []
    for key, value in values.items():
        args += ["--override", f"{key}={value}"]
    return args


def _run(module, overrides, *extra):
    assert module.main([*overrides, *extra]) == 0


def test_full_pipeline(workdir: Path, overrides: list[str]):
    from step1_data_collection import main as step1
    from step2_preprocessing import main as step2
    from step3_weak_supervision import main as step3
    from step4_model_training import cross_validation, evaluate
    from step4_model_training import main as step4
    from step5_knowledge_graph import main as step5
    from step6_cross_lingual_retrieval import main as step6

    _run(step1, overrides)
    stats = json.loads((workdir / "data/interim/collection_stats.json").read_text(encoding="utf-8"))
    assert stats["documents"] > 300

    _run(step2, overrides)
    assert (workdir / "data/processed/sentences.jsonl").stat().st_size > 0

    _run(step3, overrides)
    weak = json.loads((workdir / "data/weak_labels/weak_supervision_stats.json").read_text(encoding="utf-8"))
    entities = weak["full_corpus"]["entities"]
    assert {"DISEASE", "SYMPTOM", "DRUG", "DOSAGE", "PROCEDURE", "BODY_PART"} <= set(entities)
    assert {"TREATS", "HAS_SYMPTOM", "HAS_DOSAGE"} <= set(weak["full_corpus"]["relations"])

    _run(step4, overrides)
    model_dir = workdir / "models/km_kenet"
    metrics = json.loads((model_dir / "test_metrics.json").read_text(encoding="utf-8"))
    overall = metrics["paper_metrics"]["overall"]
    for key in ("accuracy", "precision", "recall", "specificity", "f1"):
        assert 0.0 <= overall[key] <= 1.0
    assert (model_dir / "test_results.md").is_file()

    _run(evaluate, overrides, "--split", "dev")
    assert (model_dir / "dev_metrics.json").is_file()

    _run(step5, overrides)
    kg_dir = workdir / "outputs/knowledge_graph"
    for name in ("knowledge_graph.json", "knowledge_graph.ttl", "triples.csv", "neo4j_import.cypher"):
        assert (kg_dir / name).is_file(), name

    _run(step6, overrides, "--query", "How is tuberculosis treated?")
    answers = sorted((workdir / "outputs/answers").glob("answers_*.json"))
    result = json.loads(answers[-1].read_text(encoding="utf-8"))[0]
    assert result["query_ckb"] and result["answer"]

    _run(cross_validation, overrides)
    cv = json.loads((model_dir / "cross_validation/cross_validation.json").read_text(encoding="utf-8"))
    assert len(cv["folds"]) == 2 and set(cv["mean"]) >= {"accuracy", "f1"}
