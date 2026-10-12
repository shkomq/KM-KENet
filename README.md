KM-KENet extracts medical knowledge from Central Kurdish (Sorani) text without manually annotated training data. The framework has five stages:

Data collection: loading the Kurdish Medical Corpus (KMC).
Pre-processing: cleaning, Sorani orthographic normalization and tokenization.
Weak supervision: dictionary-, pattern- and heuristic-based labeling functions whose noisy labels are aggregated into a weakly labelled dataset.
Model training: a multilingual transformer (XLM-R) fine-tuned jointly for named entity recognition and relation extraction, whose outputs form a structured medical knowledge graph.
Cross-lingual retrieval: the Aya multilingual model translates questions between English and Kurdish and answers them from the knowledge graph.
Dataset

The Kurdish Medical Corpus (KMC) V1 is publicly available under CC BY 4.0:

Hugging Face: https://huggingface.co/datasets/shkomq/Kurdish_Medical_Corpus_KMC
Mendeley Data: https://data.mendeley.com/datasets/cn5grbx63n/1
Contact

Shko Muhammed Qader — shkomq@tju.edu.cn
