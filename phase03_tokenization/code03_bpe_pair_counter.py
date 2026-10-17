from collections import Counter


def run():
    print("=" * 60)
    print("PHASE 03 - CODE 03: BPE CHARACTER PAIR COUNTING")
    print("=" * 60)

    text = "low lower lowest"
    words = text.split()

    vocab = []
    for word in words:
        chars = list(word)
        vocab.append(chars)

    print("Character Vocab Lists per Word:")
    print(vocab)

    pairs = Counter()
    for word in vocab:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += 1

    print("\nAdjacent Character Pair Frequencies:")
    print(pairs)
    print("\nMost Frequent Pair:", pairs.most_common(1)[0])


if __name__ == "__main__":
    run()
