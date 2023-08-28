"""script for preprocessing the PESV dataset.

Steps: 

- load the PESV dataset
- save each column as as separate file. This makes for faster loading times
- parse the XML column
- make training splits for finetuning
- make training splits for PET
"""

from text_classification_bibliome.preprocessing import clean_multilingual_string
from text_classification_bibliome.utils import create_write_directory
from text_classification_bibliome import pesv_preprocessing
from text_classification_bibliome.preprocessing import bibliome_basic_loading_dataframe

import pandas as pd

import os
import sys
import argparse
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
tqdm.pandas()


def separate_columns(
    pesv_df: pd.DataFrame,
    cols_dir: Union[str, os.PathLike]
):
    """Separate each column of the dataset into its onw file

    Args:
        pesv_df (pd.DataFrame): pesv dataset
        output_dir (Union[str, os.PathLike]): directory to save the columns 

    """
    create_write_directory(cols_dir)
    for ind, column_name in tqdm(enumerate(pesv_df.columns), desc="Saving columns"):
        ind = ind+1  # start at 01

        col_path = os.path.join(
            cols_dir,
            f"{column_name}.csv"
        )
        pesv_df[column_name].to_csv(
            col_path,
            encoding="utf-8",
            lineterminator="\n",
            index=False
        )


SCRAPPED_CONTENT_COLUMNS = [
    #  'trafilatura_title', # now parsed from the xml tei
    'trafilatura_extracted_xml_tei',
    "translation_title",
]
ANNOTATED_CONTENT_COLUMNS = [
    "sujet"
]

TEXT_CONTENT_COLUMNS = [
    'translation_title',
    'parsed_trafilatura_title',
    'parsed_trafilatura_abstract',
    'parsed_trafilatura_fulltext',
]


def make_reduced_pesv_dataset(all_columns_dir: str) -> pd.DataFrame:
    """Load only some columns of the PESV dataset

    Args:
        all_columns_dir (str):  directory containing all the columns of the VSI dataset

    Returns:
        pd.DataFrame: a lighter version of the PESV dataset
    """

    pesv_df = pesv_preprocessing.bibliome_pesv_build_dataframe_from_columns(
        dataset_cols_dir=all_columns_dir,
        column_names=SCRAPPED_CONTENT_COLUMNS + ANNOTATED_CONTENT_COLUMNS
    )
    return pesv_df


def make_datasets_for_text_classification(
    pesv_df: pd.DataFrame,
    datasets_dir: Union[str, os.PathLike]
):
    """The most important preprocessing function.
     This is the function that groups by text content, and assings Relevant (1) iff the text was assigned a subject at least once.
     Otherwise, it assings Irrelevant (0).



    Args:
        pesv_df (pd.DataFrame): pesv dataset
        output_dir (Union[str, os.PathLike]): directory to save the datasets 

    """
    create_write_directory(datasets_dir)
    for col in TEXT_CONTENT_COLUMNS:

        # check if subject was assigned at least once
        filtered_dataset = pesv_preprocessing.get_clean_dataset_add_has_subject_column(
            pesv_df,
            target_column=col
        )

        # save to directory
        processed_dataset_path = os.path.join(
            datasets_dir, f"{col}.csv"
        )
        print(col)
        print(f"Unique elements {len(filtered_dataset[col].unique())}")
        filtered_dataset.to_csv(
            processed_dataset_path,
            encoding="utf-8",
            lineterminator="\n"
        )


def make_finetuning_splits(
    datasets_dir: Union[str, os.PathLike],
    splits_dir: Union[str, os.PathLike]
):
    """Make splits for finetuning, both balanced and unbalanced"""

    for df_path in os.listdir(datasets_dir):
        full_df_path = os.path.join(datasets_dir, df_path)

        # making splits with unbalanced train dev
        pesv_preprocessing.split_and_save_dataset(
            input_path=full_df_path,
            output_dir=splits_dir,
            balanced_train_dev=False
        )

        # making splits with balanced train dev
        pesv_preprocessing.split_and_save_dataset(
            input_path=full_df_path,
            output_dir=splits_dir,
            balanced_train_dev=True
        )
    return


# TRAIN_DOCS_PER_CATEGORY = [50,100,200, 500, 1000]
TRAIN_DOCS_PER_CATEGORY = [1000]


def make_pet_training_splits(
    splits_dir: Union[str, os.PathLike]
):
    """Make splits for PET, for all number of examples per category"""

    balanced_datasets_path = os.path.join(splits_dir, "balanced_dev_train")

    for n_train_docs in TRAIN_DOCS_PER_CATEGORY:
        pet_split_output_dir = os.path.join(
            splits_dir,
            f"pet_{n_train_docs}"
        )
        create_write_directory(pet_split_output_dir)

        for split_dir in os.listdir(balanced_datasets_path):
            balanced_split_dir_full = os.path.join(
                balanced_datasets_path, split_dir)

            pet_split_output_dir_for_content_source = os.path.join(
                pet_split_output_dir, split_dir
            )
            create_write_directory(pet_split_output_dir_for_content_source)

            create_pet_split_from_balanced_split_dir(
                balanced_split_dir=balanced_split_dir_full,
                n_train_docs=n_train_docs,
                pet_split_output_dir=pet_split_output_dir_for_content_source
            )


def split_train_into_train_and_unlabeled(
        train_dataframe: pd.DataFrame,
        n_train_docs: int,
):
    """Take the first 2*n entries from a training split to create pet training and unlabeled splits"""
    n_entries = 2 * n_train_docs
    new_train_df = train_dataframe[:n_entries]
    unlabeled_df = train_dataframe[n_entries:]

    return new_train_df, unlabeled_df


def create_pet_split_from_balanced_split_dir(
        balanced_split_dir: str,
        n_train_docs: int,
        pet_split_output_dir: str,
):
    """Use the split with a balanced dev and trains splits to create a pet split

    Args:
        balanced_split_dir (str): location of the finetuning split with balanced dev and train
        n_train_docs (int): n docs per category
        pet_split_output_dir (str): where to save the csvs
    """

    dataset = {}
    for split in ["train", "test", "dev"]:
        dataset[split] = pd.read_csv(
            os.path.join(balanced_split_dir, split + ".csv"),
            index_col=0
        )
    new_train_df, unlabeled_df = split_train_into_train_and_unlabeled(
        dataset["train"],
        n_train_docs
    )
    dataset["train"] = new_train_df
    dataset["unlabeled"] = unlabeled_df

    for split, df in dataset.items():
        df.to_csv(
            os.path.join(pet_split_output_dir, split + ".csv")
        )


def main():

    ############################
    # CLI arguments
    ############################

    parser = argparse.ArgumentParser(
        description="Preprocess the PESV Dataset for binary classification",

    )
    parser.add_argument(
        '--pesv_dataset_path',
        default='',
        help='Path to the full VSI Dataset CSV'
    )
    parser.add_argument(
        "-o",
        "--pesv_preprocessed_path",
        default='',
        help='Directory to store the preprocessing results'
    )

    args = parser.parse_args()

    pesv_dataset_path = args.pesv_dataset_path
    output_dir = args.pesv_preprocessed_path

    ############################
    # Preprocessing
    ############################
    print("Loading full dataset")
    pesv_df = bibliome_basic_loading_dataframe(pesv_dataset_path)

    # save each raw column in its own file
    all_columns_dir = os.path.join(output_dir, "VSI_columns")
    separate_columns(pesv_df, cols_dir=all_columns_dir)

    # load only the columns we will use
    pesv_df = make_reduced_pesv_dataset(all_columns_dir)
    # parse the content from the XML
    print("Parsing XML TEI")
    pesv_df = pesv_preprocessing.add_columns_parsed_from_xml(pesv_df)

    # save the new columns
    useful_columns_dir = os.path.join(output_dir, "useful_pesv_columns")
    separate_columns(pesv_df, cols_dir=useful_columns_dir)

    # clean the columns
    print("Removing noise from strings")
    for col in TEXT_CONTENT_COLUMNS:
        print(col)
        pesv_df[col] = pesv_df[col].progress_apply(clean_multilingual_string)
        print("="*10)

    # save the cleaned columns

    cleaned_useful_columns_dir = os.path.join(
        output_dir, "cleaned_useful_pesv_columns"
    )
    separate_columns(pesv_df, cols_dir=cleaned_useful_columns_dir)

    print("Making datasets for text classification")
    # make the datasets for text classification
    datasets_dir = os.path.join(
        output_dir, "tc_datasets"
    )
    make_datasets_for_text_classification(pesv_df, datasets_dir)

    ############################
    # Splitting
    ############################
    print("Making splits")
    splits_dir = os.path.join(output_dir, "splits")
    create_write_directory(splits_dir)

    # Finetuning Splits
    make_finetuning_splits(datasets_dir, splits_dir)

    # PET splits
    make_pet_training_splits(splits_dir)
    return 0


if __name__ == "__main__":

    sys.exit(main())
