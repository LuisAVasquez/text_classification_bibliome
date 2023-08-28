"""Global utilities for Bibliome Text Classification"""

# random seed for ALL random processes in this package
import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sys
import re

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    fbeta_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import pandas as pd

RANDOM_STATE = 42

METRICS_NAMES = (
    'accuracy',
    'f1',
    'precision',
    'recall',
    "f_beta_2.0",
    "f_beta_0.5",
)


def bibliome_delete_saved_checkpoints(
    parent_dir: Union[str, os.PathLike],
    exceptions:   List[Union[str, os.PathLike]] = None,
    re_pattern: str = r"checkpoint"
) -> None:
    """Deletes all subdirectories containinig `"checkpoint"` in the name.

    Args:
        parent_dir (Union[str, os.PathLike]):
            Parent directory.

        exceptions (List[Union[str, os.PathLike]]):
            Paths to checkpoints that will be kept. Detaults to `None`.

        re_pattern (str):
            regex pattern to determine if a directory path will be deleted
            Defaults to "checkpoint". That is, all directories containing "checkpoint"
            in their (leaf) name will be deleted by default
    """
    if exceptions is None:
        exceptions = []
    exceptions = [os.path.basename(dir) for dir in exceptions]

    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
        print(f"Created output directory: {parent_dir}")

    for dir in os.listdir(parent_dir):
        if re.search(re_pattern, dir) and dir not in exceptions:
            shutil.rmtree(
                os.path.join(parent_dir, dir)
            )


def bibliome_get_dirs_for_best_and_latest_model(
        parent_dir: Union[str, os.PathLike]
) -> Tuple[str]:
    """Get the paths to the directories containing the best and latest models.

    Args:
        parent_dir (Union[str, os.PathLike]):
            Parent directory

    Returns:
        Tuple[str]:
            Tuple of the form `(best_dir_path, latest_dir_path)`
    """

    # checkpoints are saved under directories named "checkpoint-1234", etc
    checkpoint_dirs = [
        dir
        for dir in os.listdir(parent_dir)
        if "checkpoint" in dir
    ]

    checkpoint_steps = [
        int(str(dir).removeprefix("checkpoint-"))
        for dir in checkpoint_dirs
    ]

    # get directory of last checkpoint
    last_checkpoint = max(checkpoint_steps)
    last_model_checkpoint_dir = os.path.join(
        parent_dir,
        f"checkpoint-{last_checkpoint}"
    )

    # open the logs of the last checkpoint to find which was the best performing model
    with open(os.path.join(last_model_checkpoint_dir, "trainer_state.json")) as f:
        last_trainer_state = json.load(f)
    best_model_checkpoint_dir = last_trainer_state["best_model_checkpoint"]

    return (
        best_model_checkpoint_dir,
        last_model_checkpoint_dir,
    )


def bibliome_compute_metrics_sklearn(
        y_true: List,
        y_pred: List,
        average: Optional[str] = 'binary',
) -> Dict:
    """Evaluation metrics for Classification

    Args:
        y_true (List): List of true labels
        y_pred (List): List of predicted labels
        average (str, optional):
            'binary' for binary classification will give scores for the positive label (label "1")
            Use 'macro', 'micro', for multilabel classification.


    Returns:
        Dict[str, float]: Dictionary "metric name": metric value
    """

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
    )
    fbeta_2_over_1 = fbeta_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        beta=2.0  # recall weighted higher than precision
    )
    fbeta_1_over_2 = fbeta_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        beta=0.5  # precision weighted higher than recall
    )
    acc = accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'support': support,
        "f_beta_2.0": fbeta_2_over_1,
        "f_beta_0.5": fbeta_1_over_2,
    }


def bibliome_roc_curve_auc(
        y_true: List,
        probs: List,
        output_path: Union[str, os.PathLike]
) -> Dict:
    """Calculate ROC curve and AUC metric.

    Args:
        y_true (List):List of true labels
        probs (List) : List of probabilities for the positive class
        output_path (Union[str, os.PathLike]): where to save the ROC curve plot

    Returns:
        Dict: Dict with ROC curve and AUC metric.
    """

    # calculate roc curve points
    fpr, tpr, thresholds = roc_curve(
        y_true=y_true,
        y_score=probs,
    )
    auc = roc_auc_score(
        y_true=y_true,
        y_score=probs,
    )

    # plot
    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], "k--", label="Chance curve (AUC = 0.50)")
    plt.plot(fpr, tpr, label=f"Trained Classifier (AUC = {round(auc, 2)})")
    plt.legend(loc=4)
    plt.savefig(output_path)

    # convert to JSON serializable format
    fpr, tpr, thresholds = list(fpr), list(tpr), list(thresholds)

    return {
        "AUC": auc,
        "FPR": fpr,
        "TPR": tpr,
        "thresholds": thresholds,
    }


def create_write_directory(
        write_dir: Union[str, os.PathLike]
):
    """create directory. If it exists, delete it and create a new empty directory.

    Args:
        write_dir (Union[str, os.PathLike]): 
            Path to directory
    """
    # deletes previous results
    if os.path.isdir(write_dir):
        shutil.rmtree(write_dir)
    os.mkdir(write_dir)


def is_jsonable(x: Any) -> bool:
    """Check if an object can be serialized to JSON.

    Args:
        x (Any): Any python object

    Returns:
        bool: True if serializable to JSON. False if not.
    """
    try:
        json.dumps(x)
        return True
    except:
        return False


def bibliome_average_metrics_for_cv(
        metrics_list: List[Dict],
        relevant_metric: str = None,
        output_dir: Union[str, os.PathLike] = None,
) -> float:
    """Average metrics across cross validation folds. And write to a file.

    Args:
        metrics_list (List[Dict]): 
            List of dictionaries containing the metrics from cross validation

        relevant_metric (str, optional): 
            Name of the metric to be returned.
            Defaults to None. If None, will default to "f1"

    Returns:
        float: Average of the relevant metric across cross validation folds
    """

    if relevant_metric is None:
        relevant_metric = "f1"
    if relevant_metric not in METRICS_NAMES:
        raise ValueError(
            f"Averages are done on metrics supported by the package: {relevant_metric} is not supported")

    # keep only the metrics supported by the package
    metrics_list = [
        {k: v for k, v in metric_dict.items() if k in METRICS_NAMES}
        for metric_dict in metrics_list
    ]
    # calculate average and standard deviation
    averages = pd.DataFrame(metrics_list).mean().to_dict()
    standard_deviations = pd.DataFrame(metrics_list).std().to_dict()
    standard_deviations = dict(
        (f"std_{key}", val) for key, val in standard_deviations.items()
    )

    averages.update(standard_deviations)

    if output_dir is not None:
        averages_path = os.path.join(output_dir, "cv_metrics_average.json")

        with open(averages_path, "w") as f:
            json.dump(averages, f, indent=2)

    return averages[relevant_metric]

##########
# Logging utils
##########

# Log to files
# adapted from
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python


class Logger(object):
    def __init__(
            self,
            filename="Default.log",
            type="output"  # "output" or "error"
    ):
        if not os.path.isfile(filename):
            with open(filename, "w") as f:
                f.write("LOGS\n")
        if type == "output":
            self.terminal = sys.stdout
        if type == "error":
            self.terminal = sys.stderr

        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
