import os
import sentencepiece as spm


def run():
    print("=" * 60)
    print("PHASE 03 - CODE 06: SENTENCEPIECE UNIGRAM MODEL TRAINING & USAGE")
    print("=" * 60)

    training_text = """
I love artificial intelligence.
Artificial intelligence is changing software engineering.
Large language models process text using tokens.
Tokenization converts text into smaller units.
Machine learning is an important field of computer science.
Natural language processing allows computers to understand text.
Python is widely used for artificial intelligence.
Students are learning about large language models.
"""

    train_file = "training.txt"
    model_prefix = "my_tokenizer"
    model_file = f"{model_prefix}.model"

    with open(train_file, "w", encoding="utf-8") as file:
        file.write(training_text.strip())

    print(f"1. Written training text to '{train_file}'")

    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=50,
        model_type="unigram",
        character_coverage=1.0,
    )
    print(f"2. Trained SentencePiece model successfully -> '{model_file}'")

    tokenizer = spm.SentencePieceProcessor(
        model_file=model_file
    )

    text = "Artificial intelligence is changing software engineering."

    tokens = tokenizer.encode(
        text,
        out_type=str
    )

    print("\nOriginal text:")
    print(text)

    print("\nTokens (String Subwords):")
    print(tokens)

    token_ids = tokenizer.encode(
        text,
        out_type=int
    )

    print("\nToken IDs:")
    print(token_ids)

    print("\nNumber of tokens:")
    print(len(token_ids))

    decoded_text = tokenizer.decode(token_ids)

    print("\nDecoded text:")
    print(decoded_text)


if __name__ == "__main__":
    run()
