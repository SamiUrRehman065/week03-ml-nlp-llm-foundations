def run():
    print("=" * 60)
    print("PHASE 03 - CODE 10: VOCABULARY MAPPING & <UNK> TOKEN HANDLING")
    print("=" * 60)

    text = "Large language models understand text"
    tokens = text.split()

    print("Tokens:")
    print(tokens)

    vocab = {
        "<UNK>": 0,
        "Large": 1,
        "language": 2,
        "models": 3,
        "understand": 4,
        "text": 5,
    }

    print("\nVocabulary Dictionary:", vocab)

    token_ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    print("\nToken IDs:")
    print(token_ids)

    oov_text = "Large language robots understand code"
    oov_tokens = oov_text.split()
    oov_ids = [vocab.get(token, vocab["<UNK>"]) for token in oov_tokens]
    print("\nDemonstrating Out-Of-Vocabulary (<UNK>) Resolution:")
    print(f"  Sentence: '{oov_text}'")
    print(f"  Tokens:   {oov_tokens}")
    print(f"  Token IDs: {oov_ids}  (Notice 'robots' and 'code' mapped to <UNK>=0)")


if __name__ == "__main__":
    run()
