import re


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def clean_text_verbose(text: str) -> dict:
    step0_raw = text
    step1_strip = text.strip()
    step2_lower = step1_strip.lower()
    step3_spaces = re.sub(r"\s+", " ", step2_lower)
    step4_punct = re.sub(r"[^\w\s]", "", step3_spaces)

    return {
        "raw": step0_raw,
        "stripped": step1_strip,
        "lowercased": step2_lower,
        "whitespace_collapsed": step3_spaces,
        "final_cleaned": step4_punct,
    }


def run_phase02_demo():
    print("=" * 60)
    print("PHASE 02: TEXT PREPROCESSING & NORMALIZATION")
    print("=" * 60)

    original_input = "   I NEED HELP!!!   "
    original_output = clean_text(original_input)
    print(f"Original Input:  '{original_input}'")
    print(f"Cleaned Output:  '{original_output}'")

    print("\n--- MULTI-SENTENCE TEST SUITE ---")
    test_sentences = [
        "   I NEED HELP!!!   ",
        "Can ANYBODY tell me where room #402 is located???" ,
        "Special Offer!! Buy 1 GET 1 Free @ Store... Valid until 10/12/2026.",
        "Error 404:   Page not found --- please contact admin@university.edu!",
        "     Welcome     to     Natural   Language   Processing  (NLP)!!!   ",
        "Is tuition fee due on Monday, Oct 15th? Please reply ASAP!!!",
    ]

    for idx, sentence in enumerate(test_sentences, 1):
        cleaned = clean_text(sentence)
        print(f"[{idx}] Raw:     \"{sentence}\"")
        print(f"    Cleaned: \"{cleaned}\"")

    print("\n--- STEP-BY-STEP TRACE (FIRST EXAMPLE) ---")
    trace = clean_text_verbose(test_sentences[1])
    for step, val in trace.items():
        print(f"  {step.ljust(22)}: \"{val}\"")

    print("\n--- THEORETICAL SUMMARY: WHY TEXT PREPROCESSING IS USEFUL ---")
    print("1. Dimensionality Reduction: Without lowercasing, 'Course', 'course', and 'COURSE'")
    print("   would be treated as three distinct vocabulary tokens, inflating the feature matrix.")
    print("2. Noise Elimination: Punctuation (e.g., '!!!', '???', '---') often carries little semantic")
    print("   discrimination in bag-of-words tasks and fragments token matches (e.g., 'help!' vs 'help').")
    print("3. Data Consistency: Normalizing redundant whitespace ensures uniform token splitting across")
    print("   heterogeneous raw sources (web pages, user form inputs, PDFs).")
    print("4. Improved Generalization: Cleaner tokens enhance statistical overlap between training and inference data.")


if __name__ == "__main__":
    run_phase02_demo()
