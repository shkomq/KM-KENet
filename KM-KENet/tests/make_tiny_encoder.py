"""Build a tiny, randomly initialised XLM-RoBERTa encoder + tokenizer for offline tests.

The real pipeline uses FacebookAI/xlm-roberta-large from Hugging Face. For
continuous integration (and machines without internet access) this script
creates a ~1 MB model with the same architecture and API, trained-tokenizer
on the given text file, so steps 4-5 can run end to end in seconds on a CPU.
The resulting model is untrained: its scores say nothing about KM-KENet's accuracy.

    python tests/make_tiny_encoder.py --text tests/fixtures/kmc_sample.json --out /tmp/tiny-xlmr
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def build(text_file: Path, output_dir: Path, vocab_size: int = 600) -> Path:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizerFast

    text = text_file.read_text(encoding="utf-8")
    lines = [line for line in re.split(r"[\n.؟?]", text) if line.strip()]

    special = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=special, unk_token="<unk>")
    tokenizer.train_from_iterator(lines, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    fast = XLMRobertaTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
    )
    config = XLMRobertaConfig(
        vocab_size=tokenizer.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=600, pad_token_id=fast.pad_token_id,
        bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    XLMRobertaModel(config).save_pretrained(output_dir)
    fast.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"Tiny encoder saved to {build(args.text, args.out)}")


if __name__ == "__main__":
    main()
