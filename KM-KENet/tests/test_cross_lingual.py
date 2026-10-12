from common.config import load_config
from common.lexicon import load_glossary, load_lexicon
from step2_preprocessing.text_tools import build_text_tools
from step6_cross_lingual_retrieval.answer import MultilingualAssistant
from step6_cross_lingual_retrieval.aya_client import ChatClient
from step6_cross_lingual_retrieval.language import detect_language
from step6_cross_lingual_retrieval.retrievers import KGFact, detect_relation_intents
from step6_cross_lingual_retrieval.translator import LexiconTranslator


class FakeAya(ChatClient):
    """Stands in for Aya: records the prompts and returns canned replies."""

    name = "fake-aya"

    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, max_new_tokens=None):
        self.calls.append(messages)
        return "چارەسەری سیل" if "Translate" in messages[-1]["content"] else "Tuberculosis is treated with rifampicin [F1]."


def _assistant(client):
    cfg = load_config()
    tools = build_text_tools(cfg)
    translator = LexiconTranslator(load_lexicon(cfg.path("lexicon")), load_glossary(cfg.path("glossary")), tools)
    return MultilingualAssistant(client, translator)


FACT = KGFact(
    "F1", "TREATS",
    {"id": "DRUG:rifampicin", "type": "DRUG", "label_ckb": "ڕیفامپیسین", "label_en": "rifampicin"},
    {"id": "DISEASE:tuberculosis", "type": "DISEASE", "label_ckb": "سیل", "label_en": "tuberculosis"},
    support=3, confidence=0.9, evidence=[],
)


def test_language_and_intents():
    assert detect_language("How is tuberculosis treated?") == "en"
    assert detect_language("سیل چۆن چارەسەر دەکرێت؟") == "ckb"
    assert "TREATS" in detect_relation_intents("How is tuberculosis treated?")
    assert "HAS_DOSAGE" in detect_relation_intents("What is the dose of paracetamol?")


def test_aya_is_prompted_with_kg_facts():
    client = FakeAya()
    assistant = _assistant(client)
    translation, method = assistant.translate("How is tuberculosis treated?", "en", "ckb")
    assert method == "fake-aya" and translation == "چارەسەری سیل"
    answer, method = assistant.answer("How is tuberculosis treated?", [FACT], [])
    assert method == "fake-aya" and "[F1]" in answer
    assert "rifampicin" in client.calls[-1][-1]["content"]          # the KG fact is in the prompt


def test_template_answer_without_aya():
    answer, method = _assistant(None).answer("How is tuberculosis treated?", [FACT], [])
    assert method == "template" and "rifampicin" in answer and "tuberculosis" in answer
