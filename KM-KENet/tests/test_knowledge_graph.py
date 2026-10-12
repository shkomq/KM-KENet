from pathlib import Path

from common.config import load_config
from common.lexicon import load_glossary, load_lexicon
from step2_preprocessing.text_tools import build_text_tools
from step3_weak_supervision.labeling_functions.dictionary import LexiconMatcher
from step5_knowledge_graph.canonicalizer import EntityCanonicalizer
from step5_knowledge_graph.exporters import export_turtle
from step5_knowledge_graph.extractor import extract_from_weak_labels
from step5_knowledge_graph.graph_builder import KnowledgeGraphBuilder
from step6_cross_lingual_retrieval.translator import LexiconTranslator

RECORD = {
    "sentence_id": "s1", "doc_id": "d1", "text": "ئەمۆکسیسیلین بۆ چارەسەری سیل بەکاردێت",
    "tokens": ["ئەمۆکسیسیلین", "بۆ", "چارەسەری", "سیل", "بەکاردێت"],
    "entities": [{"start": 0, "end": 1, "type": "DRUG", "confidence": 0.9},
                 {"start": 3, "end": 4, "type": "DISEASE", "confidence": 0.9}],
    "relations": [{"head": 0, "tail": 1, "type": "TREATS", "confidence": 0.8}],
}


def _graph():
    cfg = load_config()
    tools = build_text_tools(cfg)
    builder = KnowledgeGraphBuilder(EntityCanonicalizer(LexiconMatcher(load_lexicon(cfg.path("lexicon")), tools), tools.stemmer))
    builder.add(extract_from_weak_labels([RECORD]))
    return builder.build(min_triple_support=1)


def test_graph_links_lexicon_entities():
    graph = _graph()
    assert graph.has_edge("DRUG:amoxicillin", "DISEASE:tuberculosis")
    assert graph.nodes["DISEASE:tuberculosis"]["label_en"] == "tuberculosis"


def test_turtle_export_is_valid_rdf(tmp_path: Path):
    import pytest

    rdflib = pytest.importorskip("rdflib")
    path = tmp_path / "kg.ttl"
    export_turtle(_graph(), path)
    graph = rdflib.Graph().parse(path, format="turtle")
    rows = list(graph.query(
        "PREFIX kmkg: <https://github.com/shkomq/SciKG-LLM/kmkg#> "
        "SELECT ?d WHERE { ?d kmkg:TREATS ?x }"
    ))
    assert len(rows) == 1


def test_lexicon_translator_english_to_kurdish():
    cfg = load_config()
    tools = build_text_tools(cfg)
    translator = LexiconTranslator(load_lexicon(cfg.path("lexicon")), load_glossary(cfg.path("glossary")), tools)
    assert "سیل" in translator.translate("How is tuberculosis treated?", "en", "ckb")
