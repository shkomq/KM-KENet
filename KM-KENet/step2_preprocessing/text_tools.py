"""Bundle of the Kurdish text tools, built once from the config and reused by later steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from common.config import PipelineConfig
from step2_preprocessing.normalizer import KurdishNormalizer, NormalizationConfig
from step2_preprocessing.stemmer import KurdishStemmer
from step2_preprocessing.tokenizer import KurdishTokenizer


@dataclass(frozen=True)
class KurdishTextTools:
    normalizer: KurdishNormalizer
    tokenizer: KurdishTokenizer = field(default_factory=KurdishTokenizer)
    stemmer: KurdishStemmer = field(default_factory=KurdishStemmer)

    def tokenize(self, text: str) -> list[str]:
        """Normalise then tokenise raw text."""
        return self.tokenizer.tokenize(self.normalizer.normalize(text))


def build_text_tools(cfg: PipelineConfig) -> KurdishTextTools:
    preprocessing = cfg.raw.get("preprocessing", {}) or {}
    normalization = NormalizationConfig.from_dict(preprocessing.get("normalization"))
    return KurdishTextTools(normalizer=KurdishNormalizer(normalization))


def load_stopwords(path: Path, tools: KurdishTextTools) -> frozenset[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                words.update(tools.tokenize(line))
    return frozenset(words)
