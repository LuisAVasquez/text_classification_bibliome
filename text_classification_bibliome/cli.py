"""Console script for text_classification_bibliome."""
from .bert_finetuning import (
    bibliome_finetune_on_dataset,
)
from .bert_cross_validation import (
    bibliome_finetune_on_dataset_with_cross_validation
)
import argparse
import sys
import time
import datetime


def main():
    """Console script for text_classification_bibliome."""
    parser = argparse.ArgumentParser(
        description="Finetune a BERT architecture for binary classification",

    )
    parser.add_argument(
        'training_arguments',
        help="Path to JSON file(s) with all arguments for finetuning. Each JSON file will trigger a new finetuning session.",
        nargs="+"
    )
    parser.add_argument(
        "-cv",
        "--cross_validation",
        action="store_true"
    )
    args = parser.parse_args()
    training_arguments_paths = args.training_arguments
    do_cross_validation = args.cross_validation

    print("Arguments: " + str(training_arguments_paths))

    # Run a finetuning session for each set of finetuning arguments
    for training_arguments_path in training_arguments_paths:
        try:
            start_time = time.time()
            if do_cross_validation:
                bibliome_finetune_on_dataset_with_cross_validation(
                    training_arguments_path
                )
            else:
                bibliome_finetune_on_dataset(training_arguments_path)
        except Exception as e:
            print(str(e))
            print("Error during training! Check logs")

        finally:
            print(f"Finish time: {datetime.datetime.now()}")
            print("--- Training duration: %s  ---" %
                  datetime.timedelta(seconds=(time.time() - start_time))
                  )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
