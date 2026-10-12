# KM-KENet — Kurdish Medical Knowledge Extraction Network

KM-KENet is a framework for extracting scientific knowledge from low-resource language corpora, applied to medical text in Central Kurdish (Sorani). It learns without manually annotated training data, builds a structured medical knowledge graph, and lets users ask questions in English or Kurdish.

## How it works

The pipeline runs in five stages:

1. **Data collection:** loads the Kurdish Medical Corpus (KMC).
2. **Pre-processing:** cleans the text, normalizes Sorani spelling and tokenizes it.
3. **Weak supervision:** dictionary-, pattern- and heuristic-based labeling functions label the corpus automatically, and their noisy labels are combined into a training dataset.
4. **Model training:** a multilingual transformer (XLM-R) is fine-tuned jointly for named entity recognition and relation extraction. Its output forms a medical knowledge graph of diseases, symptoms, drugs, dosages, procedures and body parts.
5. **Cross-lingual retrieval:** the Aya multilingual model translates questions between English and Kurdish and answers them from the knowledge graph.

## Repository contents

* Source code for each stage, with one configuration file for all settings
* Scripts to install the dependencies and run the full pipeline
* Evaluation scripts (accuracy, precision, recall, specificity, F1-score, ROC-AUC and k-fold cross-validation)
* Knowledge graph exports (JSON, RDF/Turtle for SPARQL, CSV, Neo4j)
* Automated tests that run the whole pipeline on a small sample

## Dataset

The Kurdish Medical Corpus (KMC) V1 is publicly available under CC BY 4.0:

* Hugging Face: <https://huggingface.co/datasets/shkomq/Kurdish_Medical_Corpus_KMC>
* Mendeley Data: <https://data.mendeley.com/datasets/cn5grbx63n/1>

## Contact

Shko Muhammed Qader — College of Intelligence and Computing, Tianjin University — shkomq@tju.edu.cn
