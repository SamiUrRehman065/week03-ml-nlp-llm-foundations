from transformers import AutoTokenizer


def run(tokenizer=None):
    print("=" * 60)
    print("PHASE 03 - CODE 08: CHARACTERS VS. TOKENS")
    print("=" * 60)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = """
Artificial intelligence is transforming
software engineering and many other fields.
"""

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    print("Input Text:")
    print(text.strip())
    print("\nCharacters:", len(text))
    print("Tokens:    ", len(tokens))
    print(f"Compression Ratio (Chars/Token): {len(text) / len(tokens):.2f}")


if __name__ == "__main__":
    run()
