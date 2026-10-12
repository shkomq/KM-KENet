---
language:
- ckb
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
- summarization
tags:
- medical
- kurdish
- sorani
- kmc
- nlp
- instruction-tuning
pretty_name: Kurdish Medical Corpus (KMC) V1
size_categories:
- 100K<n<1M
---

# Kurdish Medical Corpus (KMC) V1

The Kurdish Medical Corpus (KMC) V1 is a high-quality, domain-specific dataset designed for instruction-tuning and scientific knowledge discovery in Central Kurdish (Sorani). This dataset consists entirely of human-written medical content, avoiding the common pitfalls of machine-translated corpora in low-resource language research.

## Dataset Summary

KMC V1 is concentrated into a single, consolidated JSON file: `kurdish_medical_corpus_kmc.json`. It provides structured **instruction–response** pairs paired with rich document-level metadata. The content is curated from over 500 unique medical books, journals, and health portals originally written in Kurdish.

## Technical Specifications

| Metric | Value |
| :--- | :--- |
| **Total Entries** | 121,918 |
| **Total Word Count** | 6,115,043 |
| **Estimated Tokens** | ~7,949,555 |
| **Unique Source Documents** | 585 |
| **Unique Source URLs** | 205 |
| **File Format** | JSON |
| **File Size** | 132.21 MB |
| **Language** | Central Kurdish (Sorani) |
| **Region** | Kurdistan-Iraq |

## Dataset Structure

Each entry in the dataset follows this schema:

```json
{
    "id": "CKB-MED-XXXXXX",
    "instruction": "The medical question or prompt.",
    "response": "The detailed medical answer or explanation.",
    "document": {
      "title": "Source document title",
      "source_url": "Link to the original PDF/Source",
      "publication_date": "Year or date",
      "domain": "medicine"
    },
    "language": {
      "code": "ckb",
      "region": "Kurdistan-Iraq"
    }
}
```

## Top Contributing Sources

The corpus is built from a diverse set of medical literature. The top contributors (by entry count) include:

| Document Title | Count |
| :--- | :--- |
| بنچینەکانی دەرمان - general pharmacology | 6,717 |
| Principle & basics of Pharmacology | 6,399 |
| ئینسایکۆلۆپیدیای دەرمان | 4,047 |
| گەنجینەی پزیشکیی حەکیمانی کوردەواری بەشی یەکەم | 3,667 |
| کاتێ پزیشکی لێ نەبێ چاپی دووەم | 3,118 |
| طب النبوی بە کوردی | 3,116 |
| چاپی دووەم پرسیار و وەڵامی پزیشکی بۆ هەمووان | 3,017 |
| سینوهە پزیشکی تایبەتی فیرعەون بەرگی ٢ | 2,724 |
| دەرووندروستی | 2,578 |
| سینوهە پزیشکی تایبەتی فیرعەون بەرگی ١ | 2,469 |

## Research Importance

- **Low-Resource Advantage**: Addresses the critical gap in Kurdish-language medical AI.
- **Native Content**: 100% human-authored text, ensuring natural linguistic patterns and culture-specific medical context.
- **SFT Ready**: Optimized for Supervised Fine-Tuning (SFT) of Large Language Models.
- **Traceable Health Data**: Every entry preserves its source metadata, crucial for academic and clinical verification.

## Acknowledgements

This corpus was developed at the College of Intelligence and Computing, Tianjin University, China. The author, Shko Muhammed Qader, extends sincere gratitude to his supervisor, Professor Zheng Wang, for his continuous guidance, support, and insightful feedback throughout the development of this work.
The author also acknowledges the valuable contributions of colleagues and collaborators who assisted in the collection and curation of Kurdish medical resources, which were essential to the construction of this corpus. Special thanks are extended to Mr. Bashdar Rzgar Fatih for his dedicated support and assistance.

## Citation

```bibtex
@dataset{kmc_v1_2026,
  title={Kurdish Medical Corpus (KMC) V1},
  author={Shko Muhammed Qader and Zheng Wang},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/shkomq/Kurdish_Medical_Corpus_KMC}}
}
```

## Contact
For specialized research collaborations or questions, please contact the repository owner.