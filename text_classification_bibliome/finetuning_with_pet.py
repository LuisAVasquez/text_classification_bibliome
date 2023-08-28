"""Finetuning BERT models using PET (Pattern-exploiting training)"""


import logging
import os
import sys
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import time
from datetime import datetime
from datetime import timedelta
import re
import json

import torch
from datasets import Dataset

from .utils import (
    bibliome_get_dirs_for_best_and_latest_model,
    bibliome_delete_saved_checkpoints,
    create_write_directory,
    is_jsonable,
    bibliome_roc_curve_auc,
    Logger,
)

from .bert_finetuning import (
    bibliome_load_dataset_for_finetuning,
    bibliome_get_arguments_dict,
    bibliome_write_bert_report_on_dataset,
    bibliome_write_bert_training_report,
    bibliome_load_pipeline_from_pretrained
)
from easydict import EasyDict

if os.environ.get("PET_REPO"):
    PET_repo_path = os.environ.get("PET_REPO")
    print("="*10)
    print(f"The PET repository is in the directory:\n{PET_repo_path}")
    print("="*10)

    sys.path.append(PET_repo_path)
    if True:
        import pet
        from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
        from pet.utils import eq_div
        from pet.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig


else:
    print("PET repository not loaded. Initialize the environment variable 'PET_REPO'")


#########
# Utilities from the PET command line interface
#########


def load_pet_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length, verbalizer_file=args.verbalizer_file,
                              cache_dir=args.cache_dir)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.pet_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.pet_num_train_epochs, max_steps=args.pet_max_steps,
                                gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, lm_training=args.lm_training, alpha=args.alpha)

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
                              decoding_strategy=args.decoding_strategy, priming=args.priming)

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name=args.task_name,
                              label_list=args.label_list, max_seq_length=args.sc_max_seq_length,
                              verbalizer_file=args.verbalizer_file, cache_dir=args.cache_dir)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.sc_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.sc_num_train_epochs, max_steps=args.sc_max_steps,
                                temperature=args.temperature,
                                gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, use_logits=args.method != 'sequence_classifier')

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def load_ipet_config(args) -> pet.IPetConfig:
    """
    Load the iPET config from the given command line arguments.
    """
    ipet_cfg = pet.IPetConfig(generations=args.ipet_generations, logits_percentage=args.ipet_logits_percentage,
                              scale_factor=args.ipet_scale_factor, n_most_likely=args.ipet_n_most_likely)
    return ipet_cfg


#########
# Handling arguments for PET
#########

names = set()


def __setup_custom_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    names.add(name)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name)


logger = get_logger('root')


def pet_load_default_arguments() -> EasyDict:
    """Returns a dictionary-like object whose keys can also be accessed as attributes


    Returns:
        EasyDict: dictionary with the default training arguments for PET
    """

    args = EasyDict()

    # Required parameters

    # The training method to use. Either regular sequence classification, PET or iPET.
    args.method = 'pet'
    # The input data dir. Should contain the data files for the task.
    args.data_dir = None
    # The type of the pretrained language model to use.
    args.model_type = None
    # Path to the pre-trained model or shortcut name.
    args.model_name_or_path = None
    # The name of the task to train/evaluate on.
    args.task_name = None
    # The output directory where the model predictions and checkpoints will be written.
    args.output_dir = None

    # PET-specific optional parameters

    # The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' for a permuted language model like XLNet (only for PET).
    args.wrapper_type = 'mlm'
    # The ids of the PVPs to be used (only for PET).
    args.pattern_ids = [0]
    # Whether to use language modeling as auxiliary task (only for PET).
    args.lm_training = False
    # Weighting term for the auxiliary language modeling task (only for PET).
    args.alpha = 0.9999
    # Temperature used for combining PVPs (only for PET).
    args.temperature = 2
    # The path to a file to override default verbalizers (only for PET).
    args.verbalizer_file = None
    # Reduction strategy for merging predictions from multiple PET models. Select either uniform weighting (mean) or weighting based on train set accuracy (wmean).
    args.reduction = 'wmean'
    # The decoding strategy for PET with multiple masks (only for PET).
    args.decoding_strategy = 'default'
    # If set to true, no distillation is performed (only for PET).
    args.no_distillation = False
    # The number of times to repeat PET training and testing with different seeds.
    args.pet_repetitions = 3
    # The maximum total input sequence length after tokenization for PET. Sequences longer than this will be truncated, sequences shorter will be padded.
    args.pet_max_seq_length = 256
    # Batch size per GPU/CPU for PET training.
    args.pet_per_gpu_train_batch_size = 4
    # Batch size per GPU/CPU for PET evaluation.
    args.pet_per_gpu_eval_batch_size = 8
    # Batch size per GPU/CPU for auxiliary language modeling examples in PET.
    args.pet_per_gpu_unlabeled_batch_size = 4
    # Number of updates steps to accumulate before performing a backward/update pass in PET.
    args.pet_gradient_accumulation_steps = 1
    # Total number of training epochs to perform in PET.
    args.pet_num_train_epochs = 3
    # If > 0: set total number of training steps to perform in PET. Override num_train_epochs.
    args.pet_max_steps = -1

    # SequenceClassifier-specific optional parameters (also used for the final PET classifier)

    # The number of times to repeat seq. classifier training and testing with different seeds.
    args.sc_repetitions = 1
    # The maximum total input sequence length after tokenization for sequence classification. Sequences longer than this will be truncated, sequences shorter will be padded.
    args.sc_max_seq_length = 256
    # Batch size per GPU/CPU for sequence classifier training.
    args.sc_per_gpu_train_batch_size = 4
    # Batch size per GPU/CPU for sequence classifier evaluation.
    args.sc_per_gpu_eval_batch_size = 8
    # Batch size per GPU/CPU for unlabeled examples used for distillation.
    args.sc_per_gpu_unlabeled_batch_size = 4
    # Number of updates steps to accumulate before performing a backward/update pass for sequence classifier training.
    args.sc_gradient_accumulation_steps = 1
    # Total number of training epochs to perform for sequence classifier training.
    args.sc_num_train_epochs = 3
    # If > 0: set total number of training steps to perform for sequence classifier training. Override num_train_epochs.
    args.sc_max_steps = -1

    # iPET-specific optional parameters

    # The number of generations to train (only for iPET).
    args.ipet_generations = 3
    # The percentage of models to choose for annotating new training sets (only for iPET).
    args.ipet_logits_percentage = 0.25
    # The factor by which to increase the training set size per generation (only for iPET).
    args.ipet_scale_factor = 5
    # If >0, in the first generation the n_most_likely examples per label are chosen even if their predicted label is different (only for iPET).
    args.ipet_n_most_likely = -1

    # Other optional parameters

    # The total number of train examples to use, where -1 equals all examples.
    args.train_examples = -1
    # The total number of test examples to use, where -1 equals all examples.
    args.test_examples = -1
    # The total number of unlabeled examples to use, where -1 equals all examples.
    args.unlabeled_examples = -1
    # If true, train examples are not chosen randomly, but split evenly across all labels.
    args.split_examples_evenly = False
    # Where to store the pre-trained models downloaded from S3.
    args.cache_dir = ""
    # The initial learning rate for Adam.
    args.learning_rate = 1e-5
    # Weight decay if we apply some.
    args.weight_decay = 0.01
    # Epsilon for Adam optimizer.
    args.adam_epsilon = 1e-8
    # Max gradient norm.
    args.max_grad_norm = 1.0
    # Linear warmup over warmup_steps.
    args.warmup_steps = 0
    # Log every X updates steps.
    args.logging_steps = 50
    # Avoid using CUDA when available.
    args.no_cuda = False
    # Overwrite the content of the output directory.
    args.overwrite_output_dir = False
    # Random seed for initialization.
    args.seed = 42
    # Whether to perform training.
    args.do_train = False
    # Whether to perform evaluation.
    args.do_eval = False
    # Whether to use priming for evaluation.
    args.priming = False
    # Whether to perform evaluation on the dev set or the test set.
    args.eval_set = 'dev'

    return args


def bibliome_pet_generate_pet_arguments(
    model_args: Dict,
    data_args: Dict,
    tokenizer_args: Dict,
    pet_args: Dict,
) -> Dict:
    """Use the parameters used for training to generate a dictionary with
    all the parameters necessary for finetuning with PET.
    This function will combine parameters from different dictionaries into a single dictionary

    Args:
        model_args (Dict): Finetuning parameters for the model
        model_args (Dict): Finetuning parameters for the model
        model_args (Dict): Finetuning parameters for the model
        model_args (Dict): Finetuning parameters for the model

    Returns:
        Dict: Parameters for PET finetuning
    """

    # start filling all the parameters for PET with their default values
    final_pet_args = pet_load_default_arguments()

    # from the model args, we get the model name or path, and the id2label
    final_pet_args.model_name_or_path = model_args["pretrained_model_name_or_path"]
    final_pet_args.id2label = model_args["id2label"]
    final_pet_args.label2id = {
        v: k for k, v in final_pet_args.id2label.items()
    }

    # from the data args, we get a path to a directory with the split dataset
    final_pet_args.data_dir = data_args["dataframe_path"]

    # from the tokenizer args, we get the max padding length
    final_pet_args.pet_max_seq_length = tokenizer_args['max_length']
    final_pet_args.sc_max_seq_length = tokenizer_args['max_length']

    # finally, we add all the arguments from the pet args
    final_pet_args.update(pet_args)

    return final_pet_args

#########
# PET finetuning
#########


def bibliome_pet_finetuning(
        args: Dict
):
    """_summary_
    This is basically an adaptation of the CLI interface for PET into a python script.

    Args:
        args (Dict): arguments for pet finetuning
    """

    logger.info("Parameters: {}".format(args))
    logger.info(json.dumps(dict(args), indent=2))

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir, ignore_errors=True)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(
            args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(
            args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
    eval_data = load_examples(
        args.task_name, args.data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
    unlabeled_data = load_examples(
        args.task_name, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(
        args)
    ipet_cfg = load_ipet_config(args)

    # adding id2labels
    for config in [
        pet_model_cfg, pet_train_cfg,  pet_eval_cfg,
        sc_model_cfg, sc_train_cfg, sc_eval_cfg,
        ipet_cfg
    ]:
        config.id2label = args.id2label
        config.label2id = args.label2id

    if args.method == 'pet':
        pet.train_pet(
            pet_model_cfg,
            pet_train_cfg,
            pet_eval_cfg,
            sc_model_cfg,
            sc_train_cfg,
            sc_eval_cfg,
            pattern_ids=args.pattern_ids,
            output_dir=args.output_dir,
            ensemble_repetitions=args.pet_repetitions,
            final_repetitions=args.sc_repetitions,
            reduction=args.reduction,
            train_data=train_data,
            unlabeled_data=unlabeled_data,
            eval_data=eval_data,
            do_train=args.do_train,
            do_eval=args.do_eval,
            no_distillation=args.no_distillation,
            seed=args.seed
        )

    elif args.method == 'ipet':
        pet.train_ipet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, ipet_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                       pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                       ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                       reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)

    elif args.method == 'sequence_classifier':
        pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                             repetitions=args.sc_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                             eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed)

    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


#########
# Pipeline for finetuning
#########

def bibliome_pet_import_pet(
        pet_args: Dict
):

    try:
        pet_repository_path = pet_args["pet_repository"]
        sys.path.append(pet_repository_path)
        print("importing pet")
        import pet
        from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
        from pet.utils import eq_div
        from pet.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
        print("finished importing pet")
        # import log
    except Exception as e:
        print(str(e))
        raise ValueError(
            "problem importing PET. Check that 'pet_args' contaings the path to the repository")


def bibliome_pet_for_finetuning_on_dataset(
    all_args_dict_or_path: Union[Dict, str, os.PathLike]
) -> Dict:
    """Full pipeline for fine-tuning using PET

    Args:
        all_args_dict_or_path (Union[Dict, str, os.PathLike]):
            Dictionary or path to JSON file containing all arguments for
            finetuning.
            See the documentation for a reference.

    Returns:
        Dict: TODO
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

    # get hyper-parameters for tokenizing the data
    tokenizer_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "tokenizer_args"
    )["tokenizer_args"]

    # get hyper-parameters for finetuning the model with pet
    pet_finetuning_args = bibliome_get_arguments_dict(
        all_args_dict_or_path, "pet_args"
    )["pet_args"]

    ###

    # Get output dir
    output_dir = pet_finetuning_args["output_dir"]
    create_write_directory(output_dir)
    ###

    # Change log file
    # sys.stdout = Logger(os.path.join(
    #    output_dir, "ExecutionStandardOutput.log"))
    # sys.stderr = Logger(os.path.join(output_dir, "ExecutionStandardError.log"))
    ###

    # Log time
    start_time = time.time()
    print(f"Start time: {datetime.now()}")
    print(pet_finetuning_args)

    # Delete checkpoints from previous training sessions:
    print("Deleting previous checkpoints")
    if os.path.exists(output_dir):
        shutil.rmtree(pet_finetuning_args['output_dir'], ignore_errors=True)
    ###

    # load arguments for pet finetuning

    final_pet_args = bibliome_pet_generate_pet_arguments(
        model_args=model_args,
        data_args=data_args,
        tokenizer_args=tokenizer_args,
        pet_args=pet_finetuning_args
    )

    """
    get name of BERT model
    bert_model_name = model_args["pretrained_model_name_or_path"]
    """

    # load the dataset
    # for PET, this dataset will only be used for evaluation on dev and/or test splits
    print("Loading dataset for PET")
    dataset = bibliome_load_dataset_for_finetuning(**data_args)
    ###

    ###

    # import PET
    # the path to the PET directory is on the key "pet_repository" in pet_args
    # bibliome_pet_import_pet(final_pet_args) # now importing is made with an environment variable

    # finetune with PET
    print("Started finetuning with PET")
    bibliome_pet_finetuning(final_pet_args)
    print("Finished finetuning with PET")

    # get directory of final model

    last_model_checkpoint_dir = os.path.join(
        final_pet_args.output_dir, "final", "p0-i0"
    )

    finetuning_results_dict = {
        "output_dir": output_dir,
        # "best_model_checkpoint_dir": best_model_checkpoint_dir,
        "last_model_checkpoint_dir": last_model_checkpoint_dir,
        # "trainer": trainer,
        # for PET, this dataset is not tokenized nor encoded. I am just keeping the key name for consistency
        "tokenized_encoded_dataset": dataset,
    }

    print("Writing report.")
    metrics_dict = bibliome_pet_write_bert_report(
        all_args_dict_or_path,
        finetuning_results_dict,
    )

    relevant_metric = pet_finetuning_args.get("metric_for_best_model", None)
    if relevant_metric is None:
        relevant_metric = "accuracy"
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
            re_pattern=r"(p\d+-i\d+)|(final)"
        )

    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
    # Convert elapsed_time to a timedelta object
    time_delta = timedelta(seconds=elapsed_time)

    # Get the hours, minutes, and seconds from the time_delta
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds // 60) % 60
    seconds = time_delta.seconds % 60

    # Format the elapsed time as hour:minute:second
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Print the formatted elapsed time
    print(f"Elapsed time: {formatted_time}")

    return finetuning_results_dict

##########
# Classification report
##########


def bibliome_pet_write_bert_report(
    all_args_dict_or_path,
    finetuning_results_dict,
    split_for_report: str = "dev",
    dataset_for_report: Dataset = None,
) -> dict:
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

    last_model_checkpoint_dir = finetuning_results_dict["last_model_checkpoint_dir"]

    #    dataset_for_report = finetuning_results_dict["tokenized_encoded_dataset"][split_for_report]
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
        file_name="results.json"
    )
    ###

    # write report on the dev and test splits
    # print(f"Checking performance of best model on {split_for_report} split")
    # load the pipeline
    pipeline = bibliome_load_pipeline_from_pretrained(
        checkpoint_dir=last_model_checkpoint_dir,
        all_args_dict_or_path=json_args_dict,
    )

    metrics_results_dev = None
    metrics_results_test = None
    try:
        print(f"Checking performance of last model on dev split")
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
        print(f"Checking performance of last model on test split")
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
