from step2_preprocessing.cleaner import KurdishScriptFilter, TextCleaner
from step2_preprocessing.normalizer import KurdishNormalizer
from step2_preprocessing.stemmer import KurdishStemmer
from step2_preprocessing.tokenizer import KurdishTokenizer, is_number


def test_normalizer_unifies_arabic_letters_and_digits():
    normalizer = KurdishNormalizer()
    assert normalizer.normalize("كتێب") == "کتێب"            # Arabic kaf -> Kurdish keheh
    assert normalizer.normalize("ي") == "ی"                  # Arabic yeh -> Farsi yeh
    assert normalizer.normalize("٥٠٠ MG") == "500 mg"       # Arabic-Indic digits, lower-cased Latin


def test_cleaner_removes_markdown_html_and_urls():
    cleaned = TextCleaner().clean("**نەخۆشی شەکرە** <b>x</b> https://example.org [3]")
    assert "*" not in cleaned and "<b>" not in cleaned and "http" not in cleaned and "[3]" not in cleaned


def test_tokenizer_and_sentence_splitter():
    tokenizer = KurdishTokenizer()
    assert tokenizer.tokenize("پاراسیتامۆڵ 500 mg.") == ["پاراسیتامۆڵ", "500", "mg", "."]
    assert len(tokenizer.split_sentences("یەکەم. دووەم؟ سێیەم")) == 3
    assert is_number("2.5") and not is_number("mg")


def test_stemmer_strips_sorani_suffixes():
    stemmer = KurdishStemmer()
    assert "گورچیلە" in stemmer.candidates("گورچیلەکان")
    assert stemmer.stem("نەخۆشییەکان") == "نەخۆش"


def test_script_filter_accepts_sorani_rejects_english():
    script_filter = KurdishScriptFilter()
    assert script_filter.accepts("نیشانەکانی نەخۆشی شەکرە بریتین لە تینوێتی")
    assert not script_filter.accepts("This sentence is English only")
