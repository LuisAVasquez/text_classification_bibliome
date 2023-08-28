"""Finetuning BERT models. Expects all arguments to come from a JSON file."""
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

import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers import (
    TextClassificationPipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.nn import CrossEntropyLoss
from torch import tensor
from datasets import DatasetDict, Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    is_jsonable,
    bibliome_roc_curve_auc,
    Logger,
)


def bibliome_get_arguments_dict(
    dict_or_path: Union[Dict, str, os.PathLike],
    keys: Union[str, List[str]] = None,
) -> Dict:
    """Get a dictionary with the arguments for the functions of this package.

    Args:
        dict_or_path (Union[Dict, str, os.PathLike]):
            Either
            - A dictionary with the arguments. In this case, the dictionary is returned unchanged.
            - A string or path to a JSON with the arguments for training.

        keys (Union[str, List[str]]):
            Only used when the first argument is a path.
            This is the key of the JSON where specific arguments are used.
            Defaults to None.

    Returns:
        Dict: A dictionary with the arguments for the function of this package
    """
    if isinstance(dict_or_path, dict):
        return dict_or_path

    if os.path.isfile(dict_or_path) and dict_or_path.endswith(".json"):
        with open(dict_or_path) as f:
            json_dict = json.load(f)

        if keys is None:
            return json_dict
        if isinstance(keys, str):
            keys = [keys]

        args_dict = {key: json_dict[key] for key in keys}
        return args_dict

    raise ValueError("Argument must be a dictionary or a path to a json file.")


def bibliome_load_dataset_for_finetuning(
    dataframe_path: Union[str, os.PathLike] = None,
    text_column_name: str = None,
    labels_column_name: str = None,
    shuffle: bool = False,
    downsample: bool = None,
    fraction: float = 0.1,
    **kwargs,
) -> pd.DataFrame:
    """Load a dataframe and prepare it for finetuning a binary classificator.

    Args:
        dataframe_path (Union[str, os.PathLike]):
            Path to the dataframe in CSV format. Defaults to None.

        text_column_name (str):
            Original name of the column containing text.
            It will be replaced by `text` in the output of this function.
            Defaults to None.

        labels_column_name (str):
            Original name of the column containing labels.
            It will be replaced by `label` in the output of this function.
            Expects a column containing only `0` and `1`.
            Defaults to None.

        shuffle (bool):
            Whether or not to shuffle the data. Defaults to False.

        downsample (bool):
            Whether or not to downsample the data. Defaults to None.

        fraction (float):
            Float between 0.0 and 1.0 specifying how much of the data will be kept.
            Only used when `downsample = True`.
            Defaults to 0.1.


    Returns:
        pd.DataFrame:
            Dataframe with two columns, `text` and `labels`, as required for training a classifier.
    """

    # load data from path, select only the text and labels columns, and rename them.
    # shuffle if requested
    dataset = bibliome_load_dataframe(
        dataframe_path,
        columns=[text_column_name, labels_column_name],
        column_rename_scheme={
            text_column_name: "text",
            labels_column_name: "labels",
        },
        shuffle=shuffle,
    )
    # dealing with single csv files vs directories (with test, train, dev splits)
    if isinstance(dataset, dict):
        # at the end we return a dictionary
        pass
    if isinstance(dataset, pd.DataFrame):
        # we put it in a dictionary, and at the end we return the dataframe
        dataset = {"complete": dataset}

    for split_name, df in dataset.items():

        print(f"Loaded {split_name} dataset with {len(df)} entries.")

        # Drop empty entries
        df['text'].replace('', np.nan, inplace=True)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            print(
                f"Dropping {num_nan} NaN values from dataset, that is, {round(100*num_nan/len(df),2)}% of the {split_name} dataset"
            )
            df = df.dropna()
            print(f"New dataset size: {len(df)}")

        # downsample if requested
        if downsample:
            df = bibliome_downsample_dataframe(df, fraction=fraction)
        dataset[split_name] = df

    if "complete" in dataset.keys():
        return dataset['complete']
    else:
        return dataset


def biliome_preprocess_dataset_for_finetuning(
    dataset: Union[pd.DataFrame, dict] = None,
    train_size: int = 0.8,
    dev_size: int = 0.1,
    test_size: int = 0.1,
    bert_tokenizer: Union[str, os.PathLike] = None,
    tokenizer_args: Dict = None,
    **kwargs,
) -> DatasetDict:
    """Preprocess a dataset into a DatasetDict object in pytorch format.
    Tokenization is applied here.

    Args:
        dataframe (Union[pd.DataFrame, dict], optional):
            Dataframe with two columns, `text` and `labels`.
            There should be no `NaN`s in this dataframe.
            Alternatively, a dict whose values are dataframes of the type just described.
            Defaults to None.

        train_size (float, optional):
            Train split size. Defaults to 0.8.

        dev_size (float, optional):
            Dev split size. Defaults to 0.1.

        test_size (float, optional):
            Test split size. Defaults to 0.1.

        bert_tokenizer (Union[str, os.PathLike] , optional):
            Name or path of the bert tokenizer to use.
            Defaults to None.

        tokenizer_args (Dict):
            Dictionary containing arguments for the `encode` method of a BERT Tokenizer.
            (`max_length`, `padding`, `truncation`)

    Returns:
        DatasetDict: _description_
    """

    if isinstance(dataset, dict):
        split = dataset
    if isinstance(dataset, pd.DataFrame):
        # split the dataframe
        split = bibliome_test_train_dev_split(
            df=dataset,
            train=train_size,
            dev=dev_size,
            test=test_size,
            shuffle=False,  # dataset should have already been shuffled
        )

    # convert to DatasetDict
    split_dataset = bibliome_build_DatasetDict(split)

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


def bibliome_finetune_on_dataset(
    all_args_dict_or_path: Union[Dict, str, os.PathLike]
) -> Dict:
    """Full pipeline for fine-tuning.

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
    preprocessing_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "preprocessing_args"
    )["preprocessing_args"]

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
    print("Loading dataset")
    dataset = bibliome_load_dataset_for_finetuning(**data_args)
    ###

    # Preprocess the dataset
    print("Loading tokenizer")
    bert_tokenizer = bibliome_get_bert_tokenizer(
        pretrained_model_name_or_path=model_args["pretrained_model_name_or_path"],
        **tokenizer_args
    )
    preprocessing_args.update(
        {"dataset": dataset, "bert_tokenizer": bert_tokenizer}
    )

    print("Preprocessing dataset")

    tokenized_encoded_dataset = biliome_preprocess_dataset_for_finetuning(
        tokenizer_args=tokenizer_args, **preprocessing_args
    )
    ###

    # Load model

    print("Loading bert model")
    bert_model = bibliome_load_bert_model(
        model_path_or_name=bert_model_name,
        **model_args
    )

    # Finetune model
    # initialize `TrainingArguments` object
    training_arguments = TrainingArguments(**finetuning_args)

    # initialize `Trainer` object
    trainer = WeightedTrainer(
        model=bert_model,
        args=training_arguments,
        train_dataset=tokenized_encoded_dataset["train"],
        eval_dataset=tokenized_encoded_dataset["dev"],
        tokenizer=bert_tokenizer,
        compute_metrics=bibliome_compute_metrics,
    )
    trainer.calculate_class_weights()

    print("Started fine-tuning...")
    trainer.train()
    print("Finished fine-tuning")
    ###

    # Get directories of best performing and last checkpoints
    output_dir = training_arguments.output_dir
    (
        best_model_checkpoint_dir,
        last_model_checkpoint_dir,
    ) = (
        bibliome_get_dirs_for_best_and_latest_model(output_dir)
    )

    ###

    finetuning_results_dict = {
        "output_dir": output_dir,
        "best_model_checkpoint_dir": best_model_checkpoint_dir,
        "last_model_checkpoint_dir": last_model_checkpoint_dir,
        "trainer": trainer,
        "tokenized_encoded_dataset": tokenized_encoded_dataset,
    }

    print("Writing report.")
    metrics_dict = bibliome_write_bert_report(
        all_args_dict_or_path,
        finetuning_results_dict,
    )
    # average the relevant metric across the folds
    relevant_metric = finetuning_args.get("metric_for_best_model", None)
    final_score = metrics_dict[relevant_metric]
    finetuning_results_dict[relevant_metric] = final_score

    # Delete unnecessary checkpoints:
    delete_checkpoints = model_args.get(
        "delete_checkpoints",
        False
    )
    if delete_checkpoints:
        print("Deleting checkpoints")
        bibliome_delete_saved_checkpoints(
            output_dir,
        )

    return finetuning_results_dict


def bibliome_load_pipeline_from_pretrained(
    checkpoint_dir: Union[str, os.PathLike],
    top_k: int = None,
    all_args_dict_or_path: Union[str, os.PathLike] = None,


) -> TextClassificationPipeline:
    """Load a checkpoint from a fine-tuning session.

    Args:
        checkpoint_dir (Union[str, os.PathLike]):
            Directory containing checkpoint files.

        top_k (int):
            Number of top results to return.
            If None, return scores for all labels.
            Defaults to None.

    Returns:
        TextClassificationPipeline:
            A pipeline object that allows applying the model to new data.
    """ """"""

    # Load a model, a tokenizer, and initialize a pipeline with them.
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir
    )

    # initialize tokenizer
    tokenizer_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "tokenizer_args"
    )["tokenizer_args"]

    bert_tokenizer = bibliome_get_bert_tokenizer(
        pretrained_model_name_or_path=checkpoint_dir,
        **tokenizer_args
    )

    _tokenizer_args = deepcopy(tokenizer_args)
    # proxies are only used to load the AutoTokenizer, not for tokenization
    if "proxies" in _tokenizer_args:
        _tokenizer_args.pop("proxies")

    # prepare additional arguments for pipeline
    pipeline_args = {}
    pipeline_args.update(_tokenizer_args)

    pipe = TextClassificationPipeline(
        model=model, tokenizer=bert_tokenizer, top_k=top_k, **pipeline_args
    )
    return pipe


class WeightedTrainer(Trainer):
    """A HuggingFace trained that uses weighted cross entropy.
    Adapted from:
    https://huggingface.co/docs/transformers/main_classes/trainer
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        # prepare inputs and models
        accelerator = Accelerator()
        inputs, model = accelerator.prepare(
            inputs, model
        )
        inputs = inputs.to(accelerator.device)
        """

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # get weights for each class
        weight = self.class_weights
        weight = weight.to(model.device)

        # compute custom loss
        loss_fct = CrossEntropyLoss(weight=weight)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def calculate_class_weights(
        self,
    ):
        """Calculate class weights from the distribution of labels in the training data.

        Using Inverse-frequency class weigthing,

        weight of class C = 1 / frequency of samples in class C

        because what matters for binary classification is the proportion between weights (w_i/w_{i+1}),
        this is equivalent to Inverse Number of samples weighting, where,

        weight of class C = 1 / number of samples in class C


        Implementation Adapted from:
        https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras

        """
        # get labels.
        # compute_class_weights does not accept Dataset objects, but it does accept numpy lists
        y_train = self.train_dataset["labels"].numpy()

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        logging.info(msg=f"class weights: {class_weights}")

        class_weights = tensor(class_weights).float()
        self.class_weights = class_weights


##########
# Classification report
##########

def bibliome_write_bert_report(
    all_args_dict_or_path: Dict,
    finetuning_results_dict: Dict,
    split_for_report: str = "dev",
    dataset_for_report: Dataset = None,
) -> None:
    """After finetuning, write a report of the trianing process in a directory.

    Args:
        all_args_dict_or_path (Dict): 
            Dictionary of path to JSON containing the hyperparameters for training.
        finetuning_results_dict (Dict):
            Dictionary with the results of the finetuning process.
            Should contain an 'output_dir' key with a path to a directory where the report will be written.
        split_for_report (str):
            split where to evaluate the model in order to write the report

        dataset_for_report (Dataset):
            Dataset object for the report. If given, overrides split_for_report
    """

    # get arguments
    output_dir = finetuning_results_dict['output_dir']
    best_model_checkpoint_dir = finetuning_results_dict["best_model_checkpoint_dir"]
    last_model_checkpoint_dir = finetuning_results_dict["last_model_checkpoint_dir"]
    if finetuning_results_dict.get("dataset_for_report") is not None:
        dataset_for_report = finetuning_results_dict.get("dataset_for_report")
    else:
        dataset_for_report = finetuning_results_dict["tokenized_encoded_dataset"][split_for_report]
    json_args_dict = bibliome_get_arguments_dict(all_args_dict_or_path)
    ###

    # Write some fine-tuning results
    finetuning_results_dict_to_json = {}
    finetuning_results_dict_to_json["finetuning_results"] = {
        k: v for k, v in finetuning_results_dict.items() if is_jsonable(v)
    }
    finetuning_results_dict_to_json["finetuning_args"] = {
        k: v for k, v in json_args_dict.items() if is_jsonable(v)
    }
    with open(os.path.join(output_dir, "finetuning_outputs.json"), "w") as f:
        json.dump(finetuning_results_dict_to_json, f, indent=2)
    ###

    # write training report
    print("Writing training logs")
    bibliome_write_bert_training_report(
        checkpoint_dir=last_model_checkpoint_dir,
        output_dir=output_dir,
    )
    ###

    # write report on the dev and test splits
    # print(f"Checking performance of best model on {split_for_report} split")
    # load the pipeline
    pipeline = bibliome_load_pipeline_from_pretrained(
        checkpoint_dir=best_model_checkpoint_dir,
        all_args_dict_or_path=json_args_dict,
    )

    metrics_results_dev = None
    metrics_results_test = None
    try:
        print(f"Checking performance of best model on dev split")
        output_dir_dev = os.path.join(output_dir, "dev")
        create_write_directory(output_dir_dev)
        metrics_results_dev = bibliome_write_bert_report_on_dataset(
            pipeline=pipeline,
            dataset_for_report=finetuning_results_dict["tokenized_encoded_dataset"]["dev"],
            output_dir=output_dir_dev
        )
    except Exception as e:
        print(e.message, e.args)
        print("some error with evaluation on dev split")
        pass

    try:
        print(f"Checking performance of best model on test split")
        output_dir_test = os.path.join(output_dir, "test")
        create_write_directory(output_dir_test)
        metrics_results_test = bibliome_write_bert_report_on_dataset(
            pipeline=pipeline,
            dataset_for_report=finetuning_results_dict["tokenized_encoded_dataset"]["test"],
            output_dir=output_dir_test
        )
    except Exception as e:
        print(e.message, e.args)
        print("some error with evaluation on test split")
        pass

    if metrics_results_dev:
        print("using metrics on dev split!")
        metrics_results = metrics_results_dev
        return metrics_results

    if metrics_results_test:
        print("using metrics on test split!")
        metrics_results = metrics_results_test
        return metrics_results

    """ 
    metrics_results = bibliome_write_bert_report_on_dataset(
    pipeline=pipeline,
    dataset_for_report=dataset_for_report,
    output_dir=output_dir
    )
    """
    return metrics_results
    ###


def bibliome_write_bert_training_report(
    checkpoint_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    file_name: str = "trainer_state.json",
) -> None:
    """Write to a directory the logs of the training process,
    as well as plots of the evolution of different metrics.

    Args:
        checkpoint_dir (Union[str, os.PathLike]): 
            Directory of the training checkpoint containing the logs.
        output_dir (Union[str, os.PathLike]): 
            Directory where the training logs will be stored, in the subdirectory "training_logs"
        file_name (str):
            name of the file with the training history. 
            Defaults to "trainer_state.json" (from huggingface) 

    """

    # The "trainer state" file contains all the logs from the training process
    trainer_state_path = os.path.join(checkpoint_dir, file_name)
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    # create directory to store this data
    write_dir = os.path.join(output_dir, "training_logs")
    create_write_directory(write_dir)

    # copy trainer file to logs directory
    with open(os.path.join(write_dir, file_name), "w") as f:
        json.dump(trainer_state, f, indent=2)

    # directory for the plots
    plots_dir = os.path.join(write_dir, "plots")
    create_write_directory(plots_dir)

    # plots for all metrics and loss
    [
        plot_metric_evolution(
            trainer_state=trainer_state,
            metric_name=f"eval_{metric}",
            output_dir=plots_dir
        )
        for metric in METRICS_NAMES
    ]
    plot_metric_evolution(
        trainer_state=trainer_state,
        metric_name="eval_loss",
        output_dir=plots_dir
    )
    plot_metric_evolution(
        trainer_state=trainer_state,
        metric_name="loss",
        output_dir=plots_dir
    )


def get_metric_evolution(
    trainer_state: Dict,
    metric_name: str
) -> List:
    """Give a list of the values of a metric along the tranining process.

    Args:
        trainer_state (Dict): Dicitonary with the training logs
        metric_name (str): Metric to be searched in the logs

    Returns:
        List: List ot tuples (epoch, metric value)
    """

    metric_evolution = [
        (entry['epoch'], entry[metric_name])
        for entry in trainer_state['log_history']
        if (
            entry.get(metric_name)
            or
            entry.get(metric_name) == 0.0
        )
    ]

    return metric_evolution


def plot_metric_evolution(
        trainer_state: Dict,
        metric_name: str,
        output_dir: Union[str, os.PathLike],
):
    """Use the training logs to plot the evolution of a metric.

    Args:
        trainer_state (Dict): Dicitonary with the training logs
        metric_name (str): Metric to be ploted
        output_dir (Union[str, os.PathLike]): Directory where plot will be saved
    """

    metric_evolution = get_metric_evolution(trainer_state, metric_name)

    metric_name_for_plot = metric_name.replace("_", " ")
    metric_name_for_plot = metric_name_for_plot.replace("eval", "Evaluation")

    plt.figure()
    plt.scatter(*zip(*metric_evolution))
    plt.title(f'Training evolution of {metric_name_for_plot}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name_for_plot}')

    output_image_path = os.path.join(output_dir, f"{metric_name}.png")
    plt.savefig(output_image_path)


def bibliome_write_bert_report_on_dataset(
    pipeline: TextClassificationPipeline,
    dataset_for_report: List[str],
    output_dir: Union[str, os.PathLike],
) -> Dict:
    """Apply trained model on a dataset.
    Typically, the dataset should not have been seen by the model during training.
    Calculate the metrics.
    Save the results to the output directory.

    Args:
        pipeline (TextClassificationPipeline): Pipeline to be applied for classification.
        dataset_for_report (List[str]): Dataset to be used
        output_dir (Union[str, os.PathLike]): Directory where metrics will be saved.

    Returns:
        Dict: Dictionary with all the metrics on the dataset
    """
    # evaluate on the dataset, showing a progress bar
    print(datetime.now())
    print("Started getting predictions on dataset for evaluation")
    predictions_on_dataset = pipeline(list(dataset_for_report["text"]))
    predictions_on_dataset = [
        pred[0]  # get top prediction
        for pred in tqdm(predictions_on_dataset)
    ]
    print("Finished getting predictions on dataset for evaluation")

    # labels must be converted from tensor to list of ints
    labels = [int(id) for id in list(dataset_for_report["labels"])]

    # predictions must be converted from list of label names
    # to list of label ids
    predictions = [
        int(
            pipeline.model.config.label2id[item['label']]
        )
        for item in predictions_on_dataset
    ]

    # probabilities for positive label
    pos_label_name = pipeline.model.config.id2label[1]
    probabilities_for_positive_label = [
        # remember we're getting the top prediction.
        item['score']
        if item['label'] == pos_label_name
        else 1 - item['score']  # text not classified in the positive class
        for item in predictions_on_dataset
    ]

    results_df = {

        "labels": labels,
        "predictions": predictions,
        "probabilities_for_positive_label": probabilities_for_positive_label

    }
    results_df = pd.DataFrame(results_df)

    # create directory for the results.
    # deletes previous results
    write_dir = os.path.join(output_dir, "performance_on_split")
    create_write_directory(write_dir)

    predicitons_path = os.path.join(
        write_dir, "Predictions_on_split_dataset.csv")
    results_df.to_csv(predicitons_path)

    # calculating metrics:
    metrics_results = bibliome_compute_metrics_sklearn(
        y_true=labels,
        y_pred=predictions,
        average='binary',
    )

    # plot ROC curve
    ROC_plot_path = os.path.join(write_dir, "ROC_on_split.png")
    roc_results = bibliome_roc_curve_auc(
        y_true=labels,
        probs=probabilities_for_positive_label,
        output_path=ROC_plot_path,
    )
    metrics_results.update(roc_results)

    metrics_path = os.path.join(write_dir, "metrics_on_split.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics_results, f, indent=2)

    return metrics_results
