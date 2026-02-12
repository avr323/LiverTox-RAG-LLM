
# main.py

import argparse
from livertoxrag.pipeline import run_train, run_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiverTox RAG LLM")

    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--test", action="store_true", help="Run evaluation pipeline")
    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.test:
        run_test()
    else:
        print("Please provide --train or --test")
