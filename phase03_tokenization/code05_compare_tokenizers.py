from transformers import AutoTokenizer


def run():
    print("=" * 60)
    print("PHASE 03 - CODE 05: COMPARING TOKENIZERS (BERT VS. GPT-2)")
    print("=" * 60)

    text = """
Artificial intelligence is transforming
software engineering.
"""

    models = [
        "bert-base-uncased",
        "gpt2",
    ]

    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        tokens = tokenizer.tokenize(text)

        print(f"\nModel: {model_name}")
        print("Tokens:", tokens)
        print("Number of tokens:", len(token_ids))


if __name__ == "__main__":
    run()
