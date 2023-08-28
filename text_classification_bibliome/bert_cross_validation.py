"""Finetuning BERT models using stratified cross-validation. Expects all arguments to come from a JSON file."""
from .utils import RANDOM_STATE

import os
import sys
import shutil
import logging
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
from datetime import timedelta
from datetime import datetime
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments


from .bert_finetuning import (
    bibliome_get_arguments_dict,
)

from .utils import METRICS_NAMES
from .preprocessing import (
    bibliome_load_dataframe,
    bibliome_test_train_dev_split,
    bibliome_downsample_dataframe,
    bibliome_build_DatasetDict,
)
from .bert_utils import (
    bibliome_get_bert_tokenizer,
    bibliome_get_bert_preprocess_function,
    bibliome_load_bert_model,
    bibliome_compute_metrics,
    bibliome_compute_metrics_sklearn,
)
from .utils import (
    bibliome_get_dirs_for_best_and_latest_model,
    bibliome_delete_saved_checkpoints,
    create_write_directory,
    bibliome_average_metrics_for_cv,
    Logger,
)

from .bert_finetuning import (
    WeightedTrainer,
    bibliome_load_dataset_for_finetuning,
    biliome_preprocess_dataset_for_finetuning,
    bibliome_write_bert_report
)


def biliome_preprocess_dataset_for_finetuning_with_cross_validation(
    bert_tokenizer: Union[str, os.PathLike] = None,
    dataframe_train: pd.DataFrame = None,
    # dataframe_val:  pd.DataFrame = None,
    dataframe_test:  pd.DataFrame = None,
    tokenizer_args: Dict = None,
    **kwargs,
) -> DatasetDict:
    """Preprocess a dataset into a DatasetDict object in pytorch format.
    Tokenization is applied here.

    Args:


        bert_tokenizer (Union[str, os.PathLike] , optional):
            Name or path of the bert tokenizer to use.
            Defaults to None.

        dataframe_train_path (pd.DataFrame, optional):
            train split for cv

        dataframe_val_path (pd.DataFrame, optional):
            validation split for cv

        dataframe_test_path (pd.DataFrame, optional):
            test split for cv

        All Dataframes must be of two columns, `text` and `labels`.
        There should be no `NaN`s in these dataframes.

        tokenizer_args (Dict):
            Dictionary containing arguments for the `encode` method of a BERT Tokenizer.
            (`max_length`, `padding`, `truncation`)

    Returns:
        DatasetDict: Pytorch DatasetDict with train and test splits, processed by a BERT Tokeniwer
    """

    # split the dataframe and convert to DatasetDict
    split_dataset = bibliome_build_DatasetDict(
        {
            "train": dataframe_train,
            # "val": dataframe_val,
            "test": dataframe_test,
        }
    )
    # initialize tokenizer and text preprocessing function
    text_preprocess_function = bibliome_get_bert_preprocess_function(
        bert_tokenizer=bert_tokenizer, **tokenizer_args
    )

    # define a function that can be applied to a dataset row
    def dataset_preprocess_function(df_row):
        return text_preprocess_function(df_row["text"])

    # apply this function and get the dataset with columns containing
    # the tokenization result.
    tokenized_encoded_dataset = split_dataset.map(
        function=dataset_preprocess_function, batched=True
    )

    return tokenized_encoded_dataset


def bibliome_finetune_on_dataset_with_cross_validation(
    all_args_dict_or_path: Union[Dict, str, os.PathLike]
) -> Dict:
    """Full pipeline for fine-tuning with cross validation.

    Args:
        all_args_dict_or_path (Union[Dict, str, os.PathLike]):
            Dictionary or path to JSON file containing all arguments for
            finetuning.
            See the documentation for a reference.

    Returns:
        Dict:
        A dictionary containing:
            - The output directory from the finetuning session.
            - The directory to the best performing model checkpoint. Useful for applying the model to new data.
            - The directory to the last existing model checkpoint. Useful for getting training statistics.
            - The instance of the `Trainer` object created for this finetuning session.
            - The preprocessed data as a DataDict in pytorch format.
    """

    # Get all hyperparameters
    # get hyper-parameters for loading a NN model
    model_args = bibliome_get_arguments_dict(all_args_dict_or_path, "model_args")[
        "model_args"
    ]

    # get hyper-parameters for loading the data
    data_args = bibliome_get_arguments_dict(all_args_dict_or_path, "data_args")[
        "data_args"
    ]

    # get hyper-parameters for preprocessing the data
    cross_validation_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "cross_validation_args"
    )["cross_validation_args"]

    # get hyper-parameters for tokenizing the data
    tokenizer_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "tokenizer_args"
    )["tokenizer_args"]

    # get hyper-parameters for finetuning the model
    finetuning_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "finetuning_args"
    )["finetuning_args"]

    ###

    # Get output dir
    output_dir = finetuning_args["output_dir"]
    create_write_directory(output_dir)
    ###

    # Change log file
    sys.stdout = Logger(os.path.join(
        output_dir, "ExecutionStandardOutput.log"))
    sys.stderr = Logger(os.path.join(output_dir, "ExecutionStandardError.log"))
    ###

    # Log time
    start_time = time.time()
    print(f"Start time: {datetime.now()}")

    # Delete checkpoints from previous training sessions:
    print("Deleting previous checkpoints")
    bibliome_delete_saved_checkpoints(output_dir)
    ###

    # get name of BERT model
    bert_model_name = model_args["pretrained_model_name_or_path"]

    # load the dataset
    print("Loading datasets")
    cv_data_args = deepcopy(data_args)
    cv_data_args.update(
        {"dataframe_path": cross_validation_args["dataframe_train_path"]})
    dataset_train = bibliome_load_dataset_for_finetuning(
        **cv_data_args)
    cv_data_args.update(
        {"dataframe_path": cross_validation_args["dataframe_test_path"]})
    dataset_test = bibliome_load_dataset_for_finetuning(
        **cv_data_args
    )
    ###

    # Preprocess the dataset
    print("Loading tokenizer")
    bert_tokenizer = bibliome_get_bert_tokenizer(
        pretrained_model_name_or_path=model_args["pretrained_model_name_or_path"],
        **tokenizer_args
    )

    cross_validation_args.update(
        {"bert_tokenizer": bert_tokenizer}
    )

    print("Preprocessing dataset")

    tokenized_encoded_dataset = biliome_preprocess_dataset_for_finetuning_with_cross_validation(
        bert_tokenizer=bert_tokenizer,
        tokenizer_args=tokenizer_args,
        dataframe_train=dataset_train,
        dataframe_test=dataset_test,
    )
    ###

    # create folds
    n_folds = cross_validation_args["n_folds"]
    print(f"creating {n_folds} folds for cross validation")
    folds_indexes_pairs = bibliome_create_cv_folds(
        tokenized_encoded_dataset["train"], n_folds)

    # collect all metrics, so that they can be averaged later
    metrics_results = []

    print("Starting finetuning with cross validation")
    for fold, (train_dataset_idx, validation_dataset_idx) in enumerate(folds_indexes_pairs):

        fold_str = str(fold).zfill(3)
        print("="*5 + "Cross Validation" + "="*5)
        print(f"Now working in fold number {fold_str}")

        # get the folds from the indexes
        # fold_train_dataset = tokenized_encoded_dataset["train"][train_dataset_idx]
        # fold_validation_dataset = tokenized_encoded_dataset["train"][validation_dataset_idx]
        fold_train_dataset = tokenized_encoded_dataset["train"].select(
            train_dataset_idx)
        fold_validation_dataset = tokenized_encoded_dataset["train"].select(
            validation_dataset_idx)

        # Load model

        print("Loading bert model")
        bert_model = bibliome_load_bert_model(
            model_path_or_name=bert_model_name,
            **model_args
        )

        # Finetune model
        # initialize `TrainingArguments` object
        cv_finetunning_args = deepcopy(finetuning_args)
        # each fold has its own output dir
        fold_output_dir = os.path.join(output_dir, fold_str)
        cv_finetunning_args.update({"output_dir": fold_output_dir})

        training_arguments = TrainingArguments(**cv_finetunning_args)

        # initialize `Trainer` object
        trainer = WeightedTrainer(
            model=bert_model,
            args=training_arguments,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_validation_dataset,
            tokenizer=bert_tokenizer,
            compute_metrics=bibliome_compute_metrics,
        )
        trainer.calculate_class_weights()

        print("Started fine-tuning on current fold...")
        trainer.train()
        print("Finished fine-tuning on current fold")
        ###

        # Get directories of best performing and last checkpoints
        (
            best_model_checkpoint_dir,
            last_model_checkpoint_dir,
        ) = (
            bibliome_get_dirs_for_best_and_latest_model(fold_output_dir)
        )

        ###

        finetuning_results_dict = {
            "output_dir": fold_output_dir,
            "best_model_checkpoint_dir": best_model_checkpoint_dir,
            "last_model_checkpoint_dir": last_model_checkpoint_dir,
            "trainer": trainer,
            "tokenized_encoded_dataset": tokenized_encoded_dataset,
            "dataset_for_report": fold_validation_dataset,
        }

        print("Writing report.")
        metrics_results_in_fold = bibliome_write_bert_report(
            all_args_dict_or_path,
            finetuning_results_dict,
            split_for_report="val",
        )
        metrics_results.append(metrics_results_in_fold)

        # Delete unnecessary checkpoints:
        delete_checkpoints = model_args.get(
            "delete_checkpoints",
            False
        )
        if delete_checkpoints:
            print("Deleting checkpoints")
            bibliome_delete_saved_checkpoints(
                fold_output_dir,
            )

    print("Finished finetuning with cross validation")

    # average the relevant metric across the folds
    relevant_metric = finetuning_args.get("metric_for_best_model", None)
    final_score = bibliome_average_metrics_for_cv(
        metrics_results,
        relevant_metric,
        output_dir
    )

    return final_score


def bibliome_create_cv_folds(
        dataset: Dataset,
        n_folds: int = 10,
) -> List:
    kfold = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    folds = kfold.split(dataset, list(dataset['labels']))
    folds = list(folds)
    return folds
