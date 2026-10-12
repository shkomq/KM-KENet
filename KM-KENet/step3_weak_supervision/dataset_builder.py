"""Assembly of the weakly labelled dataset: sentence selection and leak-free splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

SPLITS = ("train", "dev", "test")


def select_training_sentences(
    records: Sequence[dict[str, Any]],
    negative_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Keep every sentence with entities plus `negative_ratio` x as many entity-free sentences."""
    positives = [record for record in records if record["entities"]]
    negatives = [record for record in records if not record["entities"]]
    keep = min(len(negatives), int(round(len(positives) * max(0.0, negative_ratio))))
    selected = positives + rng.sample(negatives, keep)
    rng.shuffle(selected)
    return selected


def split_by_group(
    records: Sequence[dict[str, Any]],
    ratios: Mapping[str, float],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Split so that all sentences of one source book/article land in the same split.

    Groups are shuffled and assigned greedily until each split reaches its share
    of sentences. Falls back to a sentence-level split if there are too few groups.
    """
    weights = {name: max(0.0, float(ratios.get(name, 0.0))) for name in SPLITS}
    total_weight = sum(weights.values())
    if total_weight <= 0 or weights["train"] <= 0:
        raise ValueError("Split ratios must be non-negative with a positive train share")
    shares = {name: weight / total_weight for name, weight in weights.items()}

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group"]].append(record)
    groups = sorted(by_group)
    rng.shuffle(groups)

    n_records = len(records)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    if len(groups) >= 3:
        targets = {name: shares[name] * n_records for name in ("test", "dev")}
        for group in groups:
            for name in ("test", "dev"):
                if len(splits[name]) < targets[name]:
                    splits[name].extend(by_group[group])
                    break
            else:
                splits["train"].extend(by_group[group])
        if splits["train"]:
            return splits

    shuffled = list(records)
    rng.shuffle(shuffled)
    n_test = int(round(shares["test"] * n_records))
    n_dev = int(round(shares["dev"] * n_records))
    splits = {
        "test": shuffled[:n_test],
        "dev": shuffled[n_test : n_test + n_dev],
        "train": shuffled[n_test + n_dev :],
    }
    return splits


def dataset_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entity_types: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    for record in records:
        entity_types.update(entity["type"] for entity in record["entities"])
        relation_types.update(relation["type"] for relation in record["relations"])
    return {
        "sentences": len(records),
        "sentences_with_entities": sum(1 for r in records if r["entities"]),
        "tokens": sum(len(r["tokens"]) for r in records),
        "entities": dict(entity_types.most_common()),
        "relations": dict(relation_types.most_common()),
    }
