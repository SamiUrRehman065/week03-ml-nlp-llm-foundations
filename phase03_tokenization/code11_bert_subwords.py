from transformers import AutoTokenizer


def run(tokenizer=None):
    print("=" * 60)
    print("PHASE 03 - CODE 11: BERT SUBWORD TOKENIZATION (WORDPIECE)")
    print("=" * 60)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Playing with language models."

    tokens = tokenizer.tokenize(text)

    print("Input Text:", text)
    print("Subword Tokens:", tokens)
    print("\nNote: 'playing' is decomposed into root 'play' and suffix '##ing' when unfamiliar.")
    print("The '##' prefix indicates that this subword attaches to the preceding token.")


if __name__ == "__main__":
    run()
