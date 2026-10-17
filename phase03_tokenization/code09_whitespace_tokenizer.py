def tokenize(text: str) -> list[str]:
    return text.lower().split()


def run():
    print("=" * 60)
    print("PHASE 03 - CODE 09: BASIC WHITESPACE TOKENIZATION")
    print("=" * 60)

    text = "Agentic AI systems can use tools"
    tokens = tokenize(text)

    print("Text:", text)
    print("Tokens:", tokens)
    print("Token count:", len(tokens))


if __name__ == "__main__":
    run()
