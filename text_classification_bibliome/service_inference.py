"""Tools for the inference service that applies Text Classification to documents from the PESV platform"""


"""
There must be one directory with the output of the training service, that is, the trained models.

The name of the directory, must contain:

- The content source used, either
    - trafilatura_title
    - trafilatura_abstract
    - trafilatura_fulltext
    - translated_title

- The training method used, either
    - pet_1000
    - finetuning

- The BERT model used, either
    - multilingual_bert
    - xlmroberta
"""


from .bert_finetuning import bibliome_load_pipeline_from_pretrained
from .pesv_preprocessing import to_be_kept
from .pesv_preprocessing import is_error_message
from .pesv_preprocessing import get_content_type
from .pesv_preprocessing import extract_trafilatura_fulltext_abstract_title
from .preprocessing import clean_multilingual_string
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import os
import itertools
CONTENT_SOURCES = [
    "trafilatura_title",
    "trafilatura_abstract",
    "trafilatura_fulltext",
    "translated_title",
]

TRAINING_METHODS = [
    "pet_1000",
    "finetuning",
]

BERT_MODELS = [
    "multilingual_bert",
    "xlmroberta",
]


def load_inference_pipelines(
        models_dir: Union[str, os.PathLike]
) -> dict:
    """Loads BERT models for classification

    Args:
        models_dir (Union[str, os.PathLike]): 
            A directory containing the output of the training service

    Returns:
        A dictionary with keys "content_source": models_list
        where models_list is a list of dictionaries, containing a classifier for each content source
    """

    pipelines = []

    for (content_source, bert_model_name, training_method) in itertools.product(CONTENT_SOURCES, BERT_MODELS, TRAINING_METHODS):
        for model_dir in os.listdir(models_dir):
            model_dir_name = model_dir.lower()
            if (
                content_source in model_dir_name
                and bert_model_name in model_dir_name
                and training_method in model_dir_name
            ):
                full_model_dir = os.path.join(
                    models_dir, model_dir
                )
                pipeline_dict = load_pipeline(
                    full_model_dir,
                    bert_model_name,
                    training_method,
                    content_source,
                )
                pipelines.append(pipeline_dict)

    return pipelines


def load_pipeline(
    model_dir: Union[str, os.PathLike],
    bert_model_name: str,
    training_method: str,
    content_source: str,
) -> dict:
    """Load a single model

    Args:
        model_dir (Union[str, os.PathLike]): directory containing the results of the training service for a single model
        bert_model_name (str): one of the BERT models
        training_method (str): training method used for the model
        content_source (str): the content source used to train the model

    Returns:
        dict: Dictionary containing 
            - a HuggingFace TextClassification Pipeline, 
            - the training method, and 
            - the bert model name
    """
    training_method = training_method.lower()
    model_dir_name = os.path.basename(model_dir).lower()

    dummy_loading_args = {
        "tokenizer_args": {
            "max_length": 300,
            "padding": "max_length",
            "truncation": True,
        }
    }

    output = {
        "pipeline": None,
        "training_method": training_method,
        "bert_model_name": bert_model_name,
        "content_source": content_source,
    }

    if "pet" in training_method:
        # directory with the model weights
        chekpoint_model_dir = os.path.join(model_dir, "final", "p0-i0")

    if "finetuning" in training_method:
        # load latest checkpoint
        checkpoints_steps = [int(d.removeprefix("checkpoint-"))
                             for d in os.listdir(model_dir) if "checkpoint" in d]
        if not checkpoints_steps:
            return output  # "pipeline = None" in the output
        checkpoints_steps.sort()
        latest_checkpoint = checkpoints_steps[-1]

        chekpoint_model_dir = os.path.join(
            model_dir,
            f"checkpoint-{latest_checkpoint}"
        )

    pipeline = bibliome_load_pipeline_from_pretrained(
        checkpoint_dir=chekpoint_model_dir,
        top_k=10,
        all_args_dict_or_path=dummy_loading_args
    )

    pipeline.model.config.id2label = {
        0: "irrelevant",
        1: "relevant"
    }
    pipeline.model.config.label2id = {
        "irrelevant": 0,
        "relevant": 1
    }

    output["pipeline"] = pipeline
    return output


def heuristic_module(
        content_source: str,
        text_content: str
) -> dict:
    """Applies heuristics to determine whether a text has content

    Args:
        content_source (str): one of the CONTENT_SOURCE
        text_content (str): the text to be analysed

    Returns:
        dict: A dictionary containing
        - a boolean, which is False iff we detect the text to be an error message or a parsing problem
        - a string containing the reason why the text was not kept if the boolean is False.
            If the boolean is true, the reason is None
    """

    # filter error messages
    if is_error_message(text_content):
        return {
            "kept": False,
            "reason": "Error message"
        }

    content_type = get_content_type(content_source)

    # filter parsing problems
    if not to_be_kept(content_type, text_content):
        return {
            "kept": False,
            "reason": "Parsing problem"
        }

    # everything ok

    return {
        "kept": True,
        "reason": None
    }


def classification_module(
    pipeline,
    text_content: str,
) -> dict:
    """Classification module for the inference service

    Args:
        pipeline (TextClassificationPipeline): _description_
        text_content (str): the text to be analyzed

    Returns:
        dict: Dictionary containing 
            - the prediction,
            - the probabilities for each label 
    """

    output = {
        "prediction": None,
        "prediction_id": None,
        "probabilities": None,
    }

    probabilities = pipeline(text_content)[0]

    ranked_probabilities = sorted(probabilities, key=lambda d: d['score'])
    prediction = ranked_probabilities[-1]["label"]
    prediction_id = pipeline.model.config.label2id[prediction]

    output["prediction"] = prediction
    output["prediction_id"] = prediction_id
    output["probabilities"] = probabilities

    return output


def single_inference(
        pipeline_dict: dict,
        content_source: str,
        text_content: str,
) -> dict:
    """Perform inference on a specific text.
    First apply heuristics. Then, if there is content in the string, apply a classifier

    Args:
        pipeline_dict (dict): A dictionary as produced by the `load_pipeline` function
        content_source (str): one of CONTENT_SOURCE
        text_content (str): the text for inference

    Returns:
        dict: A dictionary containing
            - the "kept" boolean flag 
            - the "reason" explanation string 
            - the prediction
            - the probabilities
            - the content source
            - the training method
            - the bert model name
    """

    # initialize the output with the information from the pipeline_dict, except the pipeline object.
    output = {}
    output.update(pipeline_dict)
    output.pop("pipeline")

    heuristics_result = heuristic_module(content_source, text_content)
    output.update(heuristics_result)

    if not heuristics_result["kept"]:
        # no content in the text
        output.update({
            "prediction": None,
            "prediction_id": None,
            "probabilities": None,
        }
        )
    else:
        # the text has content
        classification_result = classification_module(
            pipeline=pipeline_dict["pipeline"],
            text_content=text_content
        )
        output.update(classification_result)

    return output


def parsing_then_inference(
    pipeline_dicts: List[dict],
    xml_tei: str = None,
    trafilatura_title: str = None,
    trafilatura_abstract: str = None,
    trafilatura_fulltext: str = None,
    translation_title: str = None,
) -> dict:
    """_summary_

    Args:
        pipeline_dicts(List[dict]):  A List of dictionaries as produced by the `load_inference_pipelines` function
        xml_tei (str, optional): XML/TEI string downloaded from the internet. Defaults to None.
            If it is not None, we attempt to parse it and obtain the title, abstract, and fulltext.
            If the parsing is successfull, it overrides the values for
            the trafilatura_title, trafilatura_abstract, and trafilatura_fulltext.
            The translation_title is not affected by the presence ofthe xml_tei
        trafilatura_title (str, optional): Title obtained from trafilatura. Defaults to None.
        trafilatura_abstract (str, optional): Abstract obtained from trafilatura. Defaults to None.
        trafilatura_fulltext (str, optional): Fulltext obtained from trafilatura. Defaults to None.
        translation_title (str, optional): Title obtained from Google translate. Defaults to None.

    Returns:
        dict: a dictionary where the keys are the content sources and the values are the inference results
    """

    output = {}

    if xml_tei is not None:
        # parse the title, abstract and fulltext
        (trafilatura_title, trafilatura_abstract,
         trafilatura_fulltext) = extract_trafilatura_fulltext_abstract_title(xml_tei)

    contents = {
        "trafilatura_title": trafilatura_title,
        "trafilatura_abstract": trafilatura_abstract,
        "trafilatura_fulltext": trafilatura_fulltext,
        "translated_title": translation_title,
    }

    # remove noise from the strings
    contents = dict(
        (content_source, clean_multilingual_string(text_content))
        for (content_source, text_content) in contents.items()
    )

    for content_source in CONTENT_SOURCES:
        output[content_source] = []

        # if there is no content provided, skip to the next one
        text_content = contents[content_source]
        if text_content is None:
            continue

        for pipeline_dict in pipeline_dicts:
            # for a content source, we only use those models trained on that content source
            if pipeline_dict['content_source'] == content_source:
                inference_result = single_inference(
                    pipeline_dict,
                    content_source,
                    text_content
                )
                output[content_source].append(inference_result)

    return output
