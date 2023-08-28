"""Utilities for BERT models for binary text classification"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy

from .utils import RANDOM_STATE
from .utils import bibliome_compute_metrics_sklearn
from .preprocessing import clean_multilingual_string

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed
from transformers.tokenization_utils_base import BatchEncoding

from torch.nn import Softmax
from torch import argmax

from accelerate import Accelerator


##########
# HuggingFace random seed
##########
set_seed(RANDOM_STATE)

##########
# BERT tokenizers
##########


def bibliome_get_bert_tokenizer(bert_model_name: str = None, **kwargs) -> AutoTokenizer:
    """Create an instance of a BERT tokenizer

    Args:
        bert_model_name (str): name of the bert model. e.g. "bert-base-uncased"

    Returns:
        AutoTokenizer: Instance of the BERT tokenizer
    """
    return AutoTokenizer.from_pretrained(use_fast=True, **kwargs)


def bibliome_get_bert_preprocess_function(
    bert_tokenizer: AutoTokenizer,
    **tokenizer_args,
) -> Callable:
    """Function to convert text to BERT token encodings.

    Args:
        bert_tokenizer (AutoTokenizer):
            instance of a BERT tokenizer

        tokenizer_args:
            Dictionary containing arguments for the `encode` method of a BERT Tokenizer.
            (`max_length`, `padding`, `truncation`)

    Returns:
        Callable: Function that takes text an encodes it with a BERT tokenizer.

    """
    _tokenizer_args = deepcopy(tokenizer_args)
    _tokenizer_args.update({"return_tensors": "pt"}
                           )  # always return pytorch tensors
    # proxies are only used to load the AutoTokenizer, not for tokenization
    if "proxies" in _tokenizer_args:
        _tokenizer_args.pop("proxies")

    def preprocess_function(
        text_input: Union[str, List[str]], tokenizer_args=_tokenizer_args
    ) -> BatchEncoding:
        """Encode text using a BERT tokenizer

        Args:
            text_input (Union[str, List[str]]):
                String to be tokenized

        Returns:
            BatchEncoding: Encoded text from Tokenization result. (has `input_ids`, `attention_maks`, etc)
        """
        # clean the multilingual string
        # implement funcitonality for iterables of strings
        """
        if not isinstance(text_input, str):
            print(type(text_input))
            text_input = [clean_multilingual_string(
                text) for text in text_input]
        else:
            text_input = clean_multilingual_string(text_input)
        """
        return bert_tokenizer(text_input, **tokenizer_args)

    return preprocess_function


##########
# Load BERT models
##########


def bibliome_load_bert_model(
    model_path_or_name: Union[str, os.PathLike],
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[str, str]] = None,
    freeze_base_weights: bool = True,
    classifier_dropout: float = 0.3,
    **kwargs,
) -> AutoModelForSequenceClassification:
    """Load a BERT model for sequence classification. One can load a locally stored pretrained model or import one from HuggingFace.

    Args:
        model_path_or_name (Union[str, os.PathLike]): Either the path to a directory containing a pre-trained model,
        or, for a brand new model, use the name of a pretrained model as found in hugging-face, e.g. 'bert-base-uncased'

        num_labels (Optional[int], optional): Number of different labels in the dataset. Defaults to `None`.

        id2label (Optional[Dict[str,str]], optional): Mapping from label IDs to label name.
            e.g.. {"0" : "negative", "1" : positive}
            Defaults to `None`.

        freeze_base_weights (bool): Whether or not to keep the weights of the original BERT base.
        classifier_dropout (float): The dropout ration for the classification head.

    Returns:
        AutoModelForSequenceClassification: Instance of BERT model
    """
    model_kwargs = deepcopy(kwargs)
    # "delete_checkpoints is not used to load the model"
    model_kwargs.pop("delete_checkpoints")

    if os.path.isdir(model_path_or_name):
        print("Loading bert model from checkpoint path: {model_path_or_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path_or_name, **model_kwargs
        )
    else:
        print("Loading bert model from HuggingFace")
        model = AutoModelForSequenceClassification.from_pretrained(
            # model_path_or_name, # name comes from the kwargs
            num_labels=num_labels,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            classifier_dropout=classifier_dropout,
            **model_kwargs,
        )

    # Freezing the encoder layers
    if freeze_base_weights:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Activate Evaluation/Prediction mode
    model.eval()

    return model


##########
# Fine-tune BERT models
##########


def bibliome_compute_metrics(
    pred: BatchEncoding, average: Optional[str] = "binary"
) -> Dict[str, float]:
    """Evaluation metrics for classification.

    Args:
        preds (BatchEncoding): Output from a BERT tokenizer.
        i.e., the result from model(**inputs)

        average (str, optional):
            'binary' for binary classification will give scores for the positive label (label "1")
            Use 'macro', 'micro', for multilabel classification.


    Returns:
        Dict[str, float]: Dictionary "metric name": metric value
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return bibliome_compute_metrics_sklearn(
        y_true=labels, y_pred=preds, average=average
    )


##########
# Apply BERT models for new predictions
##########


def bibliome_bert_predict(
    text_input: Union[str, List[str]],
    bert_model: AutoModelForSequenceClassification,
    preprocessing_function: Callable,
) -> Dict[str, Any]:
    """Apply a bert model to classify text.

    Args:
        text_input (Union[str, List[str]]): Text to be classified
        preprocessing_function (Callable): Function from text to Bert model input
        bert_model (AutoModelForSequenceClassification) : Instance of Bert model for classification

    Returns:
        Dict[str, Any]: Dictionary with prediction, label, and probabilities
    """

    # get the input for the bert model
    if isinstance(text_input, str):
        text_input = [text_input]

    input_dict = {"text": text_input}

    encoded_inputs = preprocessing_function(input_dict)

    # prepare inputs and models
    accelerator = Accelerator()
    (encoded_inputs, bert_model) = accelerator.prepare(
        encoded_inputs, bert_model
    )
    encoded_inputs = encoded_inputs.to(accelerator.device)

    # activate evaluation mode and apply model
    bert_model.eval()
    outputs = bert_model(**encoded_inputs)

    # make predictions
    logits = outputs["logits"]
    print(logits)
    id2label = bert_model.config.id2label

    float_indexes = argmax(logits, dim=1)
    predicted_label_ids = [
        str(int(float_index))
        for float_index in float_indexes
    ]
    predicted_labels = [
        id2label[predicted_label_id] for predicted_label_id in predicted_label_ids
    ]
    # calculate explicit probabilities
    sofmax = Softmax(dim=1)
    probs_list = sofmax(logits)  # .tolist()

    result = [
        {
            "predicted_label": predicted_label,
            "predicted_label_id": predicted_label_id,
            "probabilities": {str(idx): float(prob) for idx, prob in enumerate(probs)},
            "probabilities_per_label_name": {
                id2label[str(idx)]: float(prob) for idx, prob in enumerate(probs)
            },
        }
        for predicted_label, predicted_label_id, probs, in zip(
            predicted_labels,
            predicted_label_ids,
            probs_list,
        )
    ]

    return result if len(result) > 1 else result[0]


# TODO
# maybe a "fast predictions" function? So that the user
# does not have to initialize the tokenizer and preprocessing functions

# Template for all arguments for finetuning
"""
args_dict = {
  "model_args": {
    "pretrained_model_name_or_path": "bert-base-multilingual-cased",
    "freeze_base_weights": False,
    "id2label": {
      "0": "irrelevant",
      "1": "relevant"
    },
    "num_labels": 2,
    "delete_checkpoints": True,
    "classifier_dropout": 0.3,
  },
  "data_args": {
    "dataframe_path": PATH.csv,
    "text_column_name": "trafilatura_title",
    "labels_column_name": "has_subject",
    "shuffle": True,
    "downsample": True,
  },
  "tokenizer_args": {
    "max_length": 100,
    "padding": "max_length",
    "truncation": True,
  },
  "preprocessing_args": {
    "train_size": 0.8,
    "dev_size": 0.1,
    "test_size": 0.1
  },
  "cross_validation_args": {
    "n_folds": 10,
    "dataframe_train_path": PATH.csv,
    "dataframe_test_path": PATH.csv,
  },
  "finetuning_args": {
    "output_dir": OUTPUT_DIR, 
    "num_train_epochs": 8,
    "overwrite_output_dir": True,
    "evaluation_strategy": "epoch",
    "eval_steps": 50,
    "save_strategy": "epoch",
    "save_steps": 50,
    "learning_rate": 1e-02,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "load_best_model_at_end": True,
    "logging_strategy": "epoch",
    "logging_steps": 50,
    "save_total_limit": 2,
    "disable_tqdm": False,
    "optim": "adamw_torch",
    "metric_for_best_model": "f_beta_2.0"
  }
}

"""
