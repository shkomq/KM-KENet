"""Step 3 — Weak Supervision Labeling.

Dictionary / pattern / heuristic LFs -> noisy label matrix -> label model
(Dawid–Skene) -> refinement & aggregation -> weakly labelled dataset.

Outputs in `weak_dir`:
* `weak_labels_full.jsonl`    every annotated sentence (used by step 5)
* `train/dev/test.jsonl`      training splits (step 4)
* `lf_summary.json`, `weak_supervision_stats.json`

    python -m step3_weak_supervision.main --config configs/config.yaml
"""

from __future__ import annotations

import itertools
import random
import sys

from common.cli import build_arg_parser, init_step
from common.io_utils import read_jsonl, require_file, write_json, write_jsonl
from common.lexicon import load_lexicon
from common.logging_utils import log_duration
from common.schema import ENTITY_TYPES, OUTSIDE
from step2_preprocessing.text_tools import build_text_tools, load_stopwords
from step3_weak_supervision.dataset_builder import (
    SPLITS,
    dataset_statistics,
    select_training_sentences,
    split_by_group,
)
from step3_weak_supervision.label_matrix import NoisyLabelGenerator, summarize_label_matrix
from step3_weak_supervision.label_model import apply_anchor_votes, build_label_model
from step3_weak_supervision.labeling_functions import (
    LexiconMatcher,
    build_ner_labeling_functions,
    build_relation_lfs,
)
from step3_weak_supervision.refinement import NERLabelRefiner, RelationLabeler

STEP_NAME = "step3_weak_supervision"


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 3 - weakly label entities and relations")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    section = cfg.section("weak_supervision")
    rng = random.Random(cfg.seed)

    sentences_path = require_file(cfg.path("processed_dir") / "sentences.jsonl", "Run step 2 first.")
    max_sentences = section.get("max_sentences")
    sentences = list(itertools.islice(read_jsonl(sentences_path), int(max_sentences) if max_sentences else None))
    if not sentences:
        logger.error("No sentences found in %s", sentences_path)
        return 1
    logger.info("Loaded %d sentences", len(sentences))

    # --- Labeling functions -------------------------------------------------
    tools = build_text_tools(cfg)
    lexicon = load_lexicon(cfg.path("lexicon"))
    stopwords = load_stopwords(cfg.path("stopwords"), tools)
    matcher = LexiconMatcher(lexicon, tools)
    ner_lfs = build_ner_labeling_functions(matcher, tools.stemmer, stopwords)
    logger.info("Lexicon: %d entries | stopwords: %d | token LFs: %d", len(lexicon), len(stopwords), len(ner_lfs))

    # --- Noisy label generation ----------------------------------------------
    classes = [OUTSIDE, *ENTITY_TYPES]
    with log_duration(logger, "Noisy label generation"):
        label_matrix = NoisyLabelGenerator(ner_lfs, classes).generate([s["tokens"] for s in sentences])
    weak_dir = cfg.path("weak_dir")
    lf_summary = summarize_label_matrix(label_matrix)
    write_json(weak_dir / "lf_summary.json", lf_summary)
    for lf in lf_summary["labeling_functions"]:
        logger.info(
            "  %-36s coverage=%.4f overlap=%.4f conflict=%.4f",
            lf["name"], lf["coverage"], lf["overlap"], lf["conflict"],
        )

    # --- Label aggregation -----------------------------------------------------
    label_model = build_label_model(section, len(classes), cfg.seed)
    with log_duration(logger, f"Fitting {type(label_model).__name__}"):
        label_model.fit(label_matrix.matrix)
        labels, confidences = label_model.predict(label_matrix.matrix)
    anchors = [label_matrix.lf_names.index(name) for name in section.get("anchor_lfs", []) or []
               if name in label_matrix.lf_names]
    if anchors:
        changed = apply_anchor_votes(
            label_matrix.matrix, labels, confidences, anchors,
            confidence=float(section.get("anchor_confidence", 0.95)),
        )
        logger.info("Anchor LFs %s set %d token labels", section.get("anchor_lfs"), changed)

    # --- Label refinement --------------------------------------------------------
    refiner = NERLabelRefiner(
        classes,
        stopwords,
        tools.stemmer,
        min_confidence=float(section.get("min_token_confidence", 0.5)),
        uncovered_as_outside=bool(section.get("uncovered_tokens_as_o", True)),
    )
    annotations = refiner.build_annotations(
        sentences, labels, confidences, label_matrix.covered, label_matrix.offsets
    )
    del label_matrix, labels, confidences

    propagation = section.get("propagation", {}) or {}
    if propagation.get("enabled", True):
        added = refiner.propagate(
            annotations,
            min_count=int(propagation.get("min_count", 3)),
            min_purity=float(propagation.get("min_purity", 0.85)),
            confidence_scale=float(propagation.get("confidence_scale", 0.9)),
        )
        logger.info("Label propagation added %d entity spans", added)

    relation_cfg = section.get("relations", {}) or {}
    relation_labeler = RelationLabeler(
        build_relation_lfs(int(relation_cfg.get("max_token_distance", 25))),
        min_confidence=float(relation_cfg.get("min_confidence", 0.5)),
    )
    with log_duration(logger, "Relation labeling"):
        for annotation in annotations:
            annotation.relations = relation_labeler.label(annotation)

    # --- Weakly labelled dataset -----------------------------------------------
    records = [annotation.to_record() for annotation in annotations]
    write_jsonl(weak_dir / "weak_labels_full.jsonl", records)

    selected = select_training_sentences(records, float(section.get("negative_sentence_ratio", 0.2)), rng)
    splits = split_by_group(selected, section.get("split", {"train": 0.8, "dev": 0.1, "test": 0.1}), rng)
    for name in SPLITS:
        write_jsonl(weak_dir / f"{name}.jsonl", splits[name])

    stats = {
        "label_model": type(label_model).__name__,
        "full_corpus": dataset_statistics(records),
        "splits": {name: dataset_statistics(splits[name]) for name in SPLITS},
        "relation_lf_votes": dict(sorted(relation_labeler.vote_counts.items())),
    }
    write_json(weak_dir / "weak_supervision_stats.json", stats)
    logger.info("Entities: %s", stats["full_corpus"]["entities"])
    logger.info("Relations: %s", stats["full_corpus"]["relations"])
    logger.info(
        "Splits: %s",
        {name: len(splits[name]) for name in SPLITS},
    )
    if not splits["train"]:
        logger.error("The training split is empty; the lexicon/LFs found no entities")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
