import sys
import argparse

from phase01_text_classification import naive_bayes_classifier
from phase02_text_preprocessing import clean_text
from phase03_tokenization import tokenization_suite
from phase04_classification_pipelines import (
    code01_pipeline_classifier,
    code02_model_comparisons,
)
from phase05_embeddings_similarity import semantic_similarity
from phase06_semantic_search import document_search


def run_phase_1():
    naive_bayes_classifier.run_original_phase01()
    naive_bayes_classifier.run_comprehensive_evaluation()


def run_phase_2():
    clean_text.run_phase02_demo()


def run_phase_3():
    tokenization_suite.run_all_tokenization()


def run_phase_4():
    code01_pipeline_classifier.run()
    print()
    code02_model_comparisons.run()


def run_phase_5():
    semantic_similarity.run()


def run_phase_6():
    document_search.run()


def run_all():
    print("=" * 80)
    print("STARTING FULL EXECUTION: WEEK 03 — ML, NLP & LLM FOUNDATIONS")
    print("=" * 80)

    print("\n>>> EXECUTING PHASE 01: TEXT CLASSIFICATION (NAIVE BAYES) <<<")
    run_phase_1()

    print("\n>>> EXECUTING PHASE 02: NLP TEXT PREPROCESSING <<<")
    run_phase_2()

    print("\n>>> EXECUTING PHASE 03: TOKENIZATION SUITE (CODES 01 - 11) <<<")
    run_phase_3()

    print("\n>>> EXECUTING PHASE 04: CLASSIFICATION PIPELINES & MODEL COMPARISON <<<")
    run_phase_4()

    print("\n>>> EXECUTING PHASE 05: EMBEDDINGS & SEMANTIC SIMILARITY <<<")
    run_phase_5()

    print("\n>>> EXECUTING PHASE 06: SEMANTIC SEARCH ENGINE <<<")
    run_phase_6()

    print("\n" + "=" * 80)
    print("ALL PHASES (01 - 06) COMPLETED SUCCESSFULLY WITHOUT ERRORS.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    phase_dispatch = {
        1: run_phase_1,
        2: run_phase_2,
        3: run_phase_3,
        4: run_phase_4,
        5: run_phase_5,
        6: run_phase_6,
    }

    if args.phase:
        phase_dispatch[args.phase]()
    else:
        run_all()


if __name__ == "__main__":
    main()
