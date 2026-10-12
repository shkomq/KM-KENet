"""Generate the synthetic KMC-format sample used by the tests.

The sample is NOT part of the Kurdish Medical Corpus: it is a small set of
template sentences in Central Kurdish (Sorani), written in the same JSON format
as `kurdish_medical_corpus_kmc.json` (including the /* ... */ licence header),
so the whole pipeline can be exercised without downloading the real dataset.

    python tests/make_fixture.py            # rewrites tests/fixtures/kmc_sample.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kmc_sample.json"

DISEASES = ["نەخۆشی شەکرە", "پەستانی خوێنی بەرز", "ڕەبۆ", "سیل", "تالاسیمیا", "کەمخوێنی", "تای مالتا", "لیشمانیا"]
SYMPTOMS = ["تا", "سەرئێشە", "کۆکە", "ماندوویی", "تینوێتی", "هێڵنج", "ڕشانەوە", "سکچوون", "تەنگەنەفەسی", "سەرگێژخواردن"]
DRUGS = ["ئەنسۆلین", "مێتفۆرمین", "ئەملۆدیپین", "پاراسیتامۆڵ", "ئەسپرین", "ئایبوپرۆفین", "ئەمۆکسیسیلین",
         "سالبیوتامۆڵ", "دۆکسیسایکلین", "ڕیفامپیسین"]
PROCEDURES_TREAT = ["نەشتەرگەری", "گواستنەوەی خوێن", "چارەسەری تیشکی"]
TESTS = ["پشکنینی خوێن", "پشکنینی میز", "تیشکی ئێکس", "سۆنار", "پشکنینی شەکری خوێن"]
BODY = ["دڵ", "جگەر", "گورچیلە", "مێشک", "پێست", "گەدە", "سییەکان", "چاو"]
DOSES = ["250 mg", "500 mg", "100 mg", "دوو حەب", "10 ml"]

TEMPLATES = [
    "{drug} بۆ چارەسەری {disease} بەکاردێت.",
    "{drug} بەکاردەبرێت بۆ چارەسەری {disease} لە منداڵان و گەورەکاندا.",
    "نیشانەکانی {disease} بریتین لە {sym1} و {sym2} و {sym3}.",
    "{disease} کاریگەری لەسەر {body1} و {body2} هەیە.",
    "{disease} بە {test} دەستنیشان دەکرێت.",
    "نەخۆش دەتوانێت {drug} {dose} دوو جار لە ڕۆژێکدا وەربگرێت.",
    "{disease} دەبێتە هۆی {sym1} و {sym2}.",
    "{procedure} بۆ چارەسەری {disease} بەکاردەهێنرێت.",
    "پزیشک بۆ دەستنیشانکردنی {disease} داوای {test} دەکات.",
    "پێویستە نەخۆش بە زوویی سەردانی پزیشک بکات و ئاگاداری تەندروستی خۆی بێت.",
    "خواردنی تەندروست و وەرزشکردن ڕۆڵێکی گرنگیان لە ژیانی ڕۆژانەدا هەیە.",
]

LICENCE = "/*\n * Synthetic test sample in the KMC V1 format (CC BY 4.0 schema). Not real corpus data.\n */\n"


def build(n_records: int = 400, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for index in range(n_records):
        sentences = []
        for _ in range(rng.randint(2, 4)):
            template = rng.choice(TEMPLATES)
            sym = rng.sample(SYMPTOMS, 3)
            body = rng.sample(BODY, 2)
            sentences.append(template.format(
                drug=rng.choice(DRUGS), disease=rng.choice(DISEASES), sym1=sym[0], sym2=sym[1], sym3=sym[2],
                body1=body[0], body2=body[1], test=rng.choice(TESTS), dose=rng.choice(DOSES),
                procedure=rng.choice(PROCEDURES_TREAT),
            ))
        book = f"Synthetic test book {index % 10 + 1}"
        records.append({
            "id": f"CKB-MED-TEST-{index + 1:06d}",
            "instruction": "ئەم نەخۆشییە چۆن چارەسەر دەکرێت؟",
            "response": " ".join(sentences),
            "document": {
                "title": book,
                "source_url": f"https://example.org/synthetic/{book.replace(' ', '_')}.pdf",
                "publication_date": "2025",
                "domain": "medicine",
            },
            "language": {"code": "ckb", "region": "Kurdistan-Iraq"},
        })
    # One record about the book itself, which step 1 must skip.
    records.append({
        "id": "CKB-MED-TEST-BOOKINFO",
        "instruction": "نووسەری ئەم کتێبەکە کێیە؟",
        "response": "ئەم کتێبە لە لایەن چەند پزیشکێکەوە نووسراوە بۆ هەموو خوێنەران.",
        "document": {"title": "Synthetic test book 1", "source_url": "", "publication_date": "2025", "domain": "medicine"},
        "language": {"code": "ckb", "region": "Kurdistan-Iraq"},
    })
    return records


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(LICENCE + json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {FIXTURE}")


if __name__ == "__main__":
    main()
