from transformers import AutoTokenizer


def run(tokenizer=None):
    print("=" * 60)
    print("PHASE 03 - CODE 07: SPECIAL TOKENS ([CLS], [SEP])")
    print("=" * 60)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Hello world"

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    print("Text:", text)
    print("Tokens:", tokens)
    print("Token IDs (with special tokens):", token_ids)

    print("\nSpecial Token Properties:")
    print("cls_token:   ", tokenizer.cls_token)
    print("sep_token:   ", tokenizer.sep_token)
    print("cls_token_id:", tokenizer.cls_token_id)
    print("sep_token_id:", tokenizer.sep_token_id)


if __name__ == "__main__":
    run()
