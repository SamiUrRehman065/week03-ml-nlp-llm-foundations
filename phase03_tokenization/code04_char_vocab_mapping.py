def run():
    print("=" * 60)
    print("PHASE 03 - CODE 04: CHARACTER-LEVEL TOKENIZATION & VOCABULARY")
    print("=" * 60)

    text = "hello"

    tokens = list(text)
    print("Text:", text)
    print("Character Tokens:", tokens)

    vocab = {
        "h": 0,
        "e": 1,
        "l": 2,
        "o": 3,
    }
    print("Vocabulary Map:", vocab)

    token_ids = [vocab[ch] for ch in text]
    print("Mapped Token IDs:", token_ids)


if __name__ == "__main__":
    run()
