"""Script to generate the bash scripts to be sent to a computing cluster for training BERT models"""


import os
import sys
import argparse
import random
import json
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from text_classification_bibliome.utils import create_write_directory


FINETUNING_TEMPLATE = """#!/bin/sh
#SBATCH -o {log_file}.log.out 
#SBATCH -e {log_file}.log.err
#SBATCH --gres=gpu:1


# activate virtual environment
source {virtual_environment_activate}

python \\
    {python_script} \\
    {json_arguments_dir}/* \\
    --output_dir \\
    {training_results_output_dir}

"""


PET_TEMPLATE = """#!/bin/sh
#SBATCH -o {log_file}.log.out 
#SBATCH -e {log_file}.log.err
#SBATCH --gres=gpu:1


# activate virtual environment
source {virtual_environment_activate}

export PET_REPO={pet_repo_path}

python \\
    {python_script} \\
    {json_arguments_dir}/* \\
    --output_dir \\
    {training_results_output_dir}

"""


def get_virtual_environment_activate_path():
    """_
    Returns:
        str: path to the 'activate' script for the current virtual environment
    """
    # find the python executable
    python_executable_path = sys.executable
    # get the dir with all the binaries
    binaries_dir = os.path.dirname(python_executable_path)
    activate_path = os.path.join(binaries_dir, "activate")
    return activate_path


def get_script_path():
    """get the path to this file."""
    return os.path.realpath(os.path.dirname(__file__))


def generate_json_args(
        pesv_preprocessed_path: Union[str, os.PathLike],
        new_jsons_dir: Union[str, os.PathLike]
):
    """Generates a folder with JSON arguments for training.

    Args:
        pesv_preprocessed_path (Union[str, os.PathLike]): Path to the preprocessed PESV dataset.
        new_jsons_dir (Union[str, os.PathLike]): Directory to store the new JSON files
    """

    # get json args from the package
    default_json_args_path = os.path.join(
        get_script_path(),
        "json_args_best_classifiers"
    )

    # the JSONs for pet are inside a "petN" directory, the and similar for the JSONs for balanced finetuning
    for training_method in os.listdir(default_json_args_path):

        # create dir for new JSONS for this training method
        training_method_jsons_dir = os.path.join(
            new_jsons_dir, training_method)
        create_write_directory(training_method_jsons_dir)

        # load default jsons and use them to create the new jsons.
        # then, save the new jsons
        for json_path in os.listdir(os.path.join(default_json_args_path, training_method)):

            if "pet" in training_method.lower():
                split_dir = training_method
            if "finetuning_balanced" in training_method.lower():
                split_dir = "balanced_dev_train"
            if "finetuning_unbalanced" in training_method.lower():
                split_dir = "unbalanced"

            json_path_full = os.path.join(
                default_json_args_path, training_method, json_path
            )
            with open(json_path_full, "r") as f:
                json_args = json.load(f)

            # get the column of the PESV dataset
            content_column = json_args["data_args"]["text_column_name"]

            # set the dataset path
            dataset_path = os.path.join(
                pesv_preprocessed_path,
                "splits",
                split_dir,
                content_column
            )
            json_args["data_args"]["dataframe_path"] = dataset_path

            # save the new json
            new_json_path = os.path.join(training_method_jsons_dir, json_path)

            with open(new_json_path, "w") as f:
                json.dump(json_args, f, indent=2)


def generate_and_save_bash_scripts(
    bash_scripts_dir: str,
    training_results_dir: str = None,
    new_jsons_dir: str = None,
    logs_dir: str = None,
    pet_repo_path: str = None,
) -> str:

    # get the path to the 'activate' for the virtual environment
    virtual_environment_activate = get_virtual_environment_activate_path()

    # we get the training methods from the directory with the json arguments.
    # the JSONs for pet are inside a "petN" directory, the and similar for the JSONs for balanced finetuning
    for training_method in os.listdir(new_jsons_dir):

        # get similar arguments
        json_arguments_dir = os.path.join(new_jsons_dir, training_method)
        log_file = os.path.join(logs_dir, training_method)
        training_results_output_dir = os.path.join(
            training_results_dir, training_method
        )
        create_write_directory(training_results_output_dir)

        # get different arguments
        if "pet" in training_method.lower():
            # bash_script = PET_TEMPLATE
            # bash_script = bash_script.format(pet_repo_path=pet_repo_path)

            python_script = os.path.join(
                get_script_path(), "pet_finetuning.py"
            )

            bash_script = f"""
            #!/bin/sh
            #SBATCH -o {log_file}.log.out 
            #SBATCH -e {log_file}.log.err
            #SBATCH --gres=gpu:1


            # activate virtual environment
            source {virtual_environment_activate}

            export PET_REPO={pet_repo_path}

            python \\
                {python_script} \\
                {json_arguments_dir}/* \\
                --output_dir \\
                {training_results_output_dir}

            """

        if "finetuning" in training_method.lower():
            # bash_script = FINETUNING_TEMPLATE

            python_script = os.path.join(
                get_script_path(), "finetuning.py"
            )

            bash_script = f"""
            #!/bin/sh
            #SBATCH -o {log_file}.log.out 
            #SBATCH -e {log_file}.log.err
            #SBATCH --gres=gpu:1


            # activate virtual environment
            source {virtual_environment_activate}

            python \\
                {python_script} \\
                {json_arguments_dir}/* \\
                --output_dir \\
                {training_results_output_dir}

            """

        # eliminate leading whitespace to the left of the strings
        bash_script = textwrap.dedent(bash_script)
        bash_script = bash_script.strip()

        # save the bash script
        bash_script_path = os.path.join(
            bash_scripts_dir, f"train_with_{training_method}.sh")

        with open(bash_script_path, "w") as f:
            f.write(bash_script)


def main():

    ############################
    # CLI arguments
    ############################

    parser = argparse.ArgumentParser(
        description="Generate BASH scripts for training bert models",

    )
    parser.add_argument(
        '--pesv_preprocessed_path',
        default='',
        help='Path to the preprocessed PESV dataset. It is the same as the `output_dir` for the script preprocess_dataset.py'
    )
    parser.add_argument(
        '--pet_repo_path',
        default='',
        help='Path to the cloned PET Repo'
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default='',
        help='Directory to store the scripts and training results'
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    pet_repo_path = args.pet_repo_path
    pesv_preprocessed_path = args.pesv_preprocessed_path

    ############################
    # Empty the output directory
    ############################
    create_write_directory(output_dir)

    ############################
    # Generate JSONs with hyperparameters
    ############################

    # create  directory to store new JSONS
    new_jsons_dir = os.path.join(output_dir, "json_hyperparameters")
    create_write_directory(new_jsons_dir)
    generate_json_args(pesv_preprocessed_path, new_jsons_dir)

    ############################
    # Generate folder for training logs
    ############################
    logs_dir = os.path.join(output_dir, "training_logs")
    create_write_directory(logs_dir)

    ############################
    # Generate folder for training results
    ############################
    training_results_dir = os.path.join(output_dir, "training_results")
    create_write_directory(training_results_dir)

    ############################
    # Generate folder for bash scripts
    ############################
    bash_scripts_dir = os.path.join(output_dir, "bash_scripts")
    create_write_directory(bash_scripts_dir)

    ############################
    # Generate and save bash scripts
    ############################

    generate_and_save_bash_scripts(
        bash_scripts_dir=bash_scripts_dir,
        training_results_dir=training_results_dir,
        new_jsons_dir=new_jsons_dir,
        logs_dir=logs_dir,
        pet_repo_path=pet_repo_path,
    )
    return 0


if __name__ == "__main__":

    sys.exit(main())
