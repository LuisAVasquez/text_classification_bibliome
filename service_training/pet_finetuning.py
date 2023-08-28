#!/usr/bin/env python
"""Testing optuna for coordinate descent for text classification"""
import os
import sys
import json
import logging
import argparse
import time
import datetime

# import optuna
from text_classification_bibliome.bert_finetuning import (
    bibliome_finetune_on_dataset,
    bibliome_get_arguments_dict,
)
from text_classification_bibliome.bert_cross_validation import (
    bibliome_finetune_on_dataset_with_cross_validation
)

from text_classification_bibliome.finetuning_with_pet import (
    bibliome_pet_for_finetuning_on_dataset
)


def main():
    """Console script for coordinate descent."""
    parser = argparse.ArgumentParser(
        description="Use PET Finetune a BERT architecture for binary classification",

    )
    parser.add_argument(
        'training_arguments',
        help="Path to JSON file(s) with all arguments for finetuning. Each JSON file will trigger a new finetuning session.",
        nargs="+"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default='',
    )

    args = parser.parse_args()
    training_arguments_paths = args.training_arguments
    training_output_dir = args.output_dir

    logging.warning("Arguments: " + str(training_arguments_paths))

    # Run a finetuning session for each set of finetuning arguments
    for ind, training_arguments_path in enumerate(training_arguments_paths):
        print("="*10)
        print("\n"*5)
        # logging time
        start_time = time.time()
        print(f"Start time: {start_time}")

        if len(training_arguments_paths) > 1:
            suffix = str(ind).zfill(2)+"_"
        else:
            suffix = ""
        logging.warning(f"The suffix is : {suffix}")
        # load finetuning arguments
        args_dict = bibliome_get_arguments_dict(training_arguments_path)

        output_dir = args_dict['pet_args']['output_dir']
        output_dir = os.path.join(training_output_dir, output_dir)
        logging.warning("model output_dir: " + str(output_dir))
        args_dict['pet_args']['output_dir'] = output_dir

        logging.warning(json.dumps(args_dict, indent=2))

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        bibliome_pet_for_finetuning_on_dataset(
            args_dict
        )

        print(f"Finish time: {datetime.datetime.now()}")
        print("--- Training duration: %s  ---" %
              datetime.timedelta(seconds=(time.time() - start_time))
              )
    print("\n\n\n\n\nFINISHED ALL!!!!")
    return 0


if __name__ == "__main__":

    # Let us minimize the objective function above.
    sys.exit(main())
