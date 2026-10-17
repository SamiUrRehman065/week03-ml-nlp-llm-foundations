from phase03_tokenization import (
    code01_bert_tokens,
    code02_bert_special_tokens,
    code03_bpe_pair_counter,
    code04_char_vocab_mapping,
    code05_compare_tokenizers,
    code06_sentencepiece_train,
    code07_special_tokens_info,
    code08_char_vs_tokens,
    code09_whitespace_tokenizer,
    code10_vocab_unk_handling,
    code11_bert_subwords,
)


def run_all_tokenization():
    print("*" * 75)
    print("PHASE 03: COMPLETE TOKENIZATION EXPERIMENTAL SUITE")
    print("*" * 75)

    code01_bert_tokens.run()
    print()
    code02_bert_special_tokens.run()
    print()
    code03_bpe_pair_counter.run()
    print()
    code04_char_vocab_mapping.run()
    print()
    code05_compare_tokenizers.run()
    print()
    code06_sentencepiece_train.run()
    print()
    code07_special_tokens_info.run()
    print()
    code08_char_vs_tokens.run()
    print()
    code09_whitespace_tokenizer.run()
    print()
    code10_vocab_unk_handling.run()
    print()
    code11_bert_subwords.run()

    print("\n" + "=" * 60)
    print("--- THEORETICAL SUMMARY: SPECIAL TOKENS & SUBWORD TOKENIZATION ---")
    print("=" * 60)
    print("1. Special Tokens:")
    print("   - [CLS] (Classification Token): Added at the beginning of input sequence.")
    print("     Its hidden state vector is traditionally pooled for sequence classification tasks.")
    print("   - [SEP] (Separator Token): Marks boundaries between sentences (e.g., question & context).")
    print("   - [PAD] / [MASK] / [UNK]: Used for batch alignment, masked language modeling,")
    print("     and out-of-vocabulary fallback.")
    print("\n2. Subword Tokenization (WordPiece / BPE / Unigram):")
    print("   - Overcomes Out-Of-Vocabulary (OOV) limitations by decomposing rare or inflected words")
    print("     into common morphological sub-units (e.g., 'playing' -> 'play' + '##ing').")
    print("   - Balances vocabulary compactness (typically 30,000-50,000 tokens) with representation power,")
    print("     preventing the vocabulary explosion of word-level models while preserving semantic")
    print("     meaning better than purely character-level models.")


if __name__ == "__main__":
    run_all_tokenization()
