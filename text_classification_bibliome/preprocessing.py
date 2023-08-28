"""Text preprocessing for classification"""

import sys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re
import unicodedata
import string
import csv
from collections import Counter

import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .utils import RANDOM_STATE


def bibliome_load_dataframe(
    path: Union[str, os.PathLike],
    columns: Optional[List[str]] = None,
    column_rename_scheme: Optional[Dict[str, str]] = None,
    shuffle: bool = False,
) -> Union[pd.DataFrame, dict]:
    """Load a CSV file into a dataframe.
    If the path is a directory with train, test, and dev splits, return a dictionary with dataframe values
    The dataset may contain text in several languages.

    Args:
        path (Union[str, os.PathLike]): 
            Path to the CSV.
            If it is a directory, it should have train, test, and dev splits inside as CSVs.

        columns (Optional[List[str]], optional): 
            Columns to keep from the CSV. Defaults to `None`. 
            If None, all columns are kept.

        column_rename_scheme (Optional[Dict[str, str]], optional): _description_. Defaults to `None`.
            Keys are column names in the original dataset. Values are new column names. 
            This is needed because training BERT classifiers with HuggingFace requires a column named `label`.

        shuffle (bool):
            Whether or not to shuffle the dataframe. Defaults to `True`. Uses a constant random seed.

    Returns:
        Union[pd.DataFrame, dict]:
            Loaded dataset. Read using UTF-8.

    """
    # dealing with a single CSV file
    if os.path.isfile(path):

        # basic loading
        df = bibliome_basic_loading_dataframe(
            path,
            columns,
            column_rename_scheme,
            shuffle,
        )
        return df
    # dealing with a directory with train, test, dev splits
    if os.path.isdir(path):
        # dictionary for the splits
        splits_dict = {}

        # look through all files in directory
        for child_file in os.listdir(path):
            df_path = os.path.join(path, child_file)
            if "train" in child_file:
                splits_dict["train"] = bibliome_basic_loading_dataframe(
                    df_path,
                    columns,
                    column_rename_scheme,
                    shuffle,
                )
            if "test" in child_file:
                splits_dict["test"] = bibliome_basic_loading_dataframe(
                    df_path,
                    columns,
                    column_rename_scheme,
                    shuffle,
                )
            if "dev" in child_file:
                splits_dict["dev"] = bibliome_basic_loading_dataframe(
                    df_path,
                    columns,
                    column_rename_scheme,
                    shuffle,
                )

        return splits_dict

    raise ValueError(
        f"The given dataset path apparently does not exist : {path}")


def bibliome_basic_loading_dataframe(
        path: Union[str, os.PathLike],
        columns: Optional[List[str]] = None,
        column_rename_scheme: Optional[Dict[str, str]] = None,
        shuffle: bool = False,
) -> pd.DataFrame:
    """Load a CSV file into a dataframe.
    The CSV file may contain text in several languages.

    Args:
        path (Union[str, os.PathLike]): 
        pd.DataFrame:
            Loaded dataframe. Read using UTF-8
    """
    # basic loading
    df = pd.read_csv(
        path,
        # index_col=0,
        encoding="utf-8",
        lineterminator="\n",
    )

    # keep only specified columns
    if columns:
        df = df[columns]

    # shuffle
    if shuffle:
        df = df.sample(frac=1, random_state=RANDOM_STATE, ignore_index=True)

    # rename columns
    if column_rename_scheme:
        df.rename(
            columns=column_rename_scheme,
            inplace=True
        )
    return df


def bibliome_downsample_dataframe(
    df: pd.DataFrame,
    fraction: float = 0.1,
) -> pd.DataFrame:
    """Downsample a datafame. By default, reduce to 10% of the dataset.

    Args:
        df (pd.DataFrame): dataframe to be downsampled.
        fraction (float, optional): Target fraction for downsampling. Defaults to `0.1`.


    Returns:
        pd.DataFrame: Downsampled dataframe.
    """
    df = df.sample(
        frac=fraction,
        random_state=RANDOM_STATE
    )
    return df


def bibliome_test_train_dev_split(
    df: pd.DataFrame,
    train: float = 0.8,
    dev: float = 0.1,
    test: float = 0.1,
    shuffle: bool = False,
    save_dir: Union[str, os.PathLike] = None,
    balanced_train_dev: bool = False,
    content_column: str = 'text',
    labels_column: str = 'labels',
) -> Dict[str, pd.DataFrame]:
    """Split a dataset into train, dev, and test splits. 
    The fractions must add up to one.
    The split is stratified, except when `balanced_train_dev = True`
    Returns a dictionary with the three splits.

    Args:
        df (pd.DataFrame): Dataframe to be split
        train (float, optional): Train split size. Defaults to 0.8.
        dev (float, optional): Dev split size. Defaults to 0.1.
        test (float, optional): Test split size. Defaults to 0.1.
        shuffle (bool, optional): Whether or not to shuffle the Dataframe. Defaults to False.
        save_dir (Union[str, os.PathLike], optional): Directory for saving the splits. Defaults to None.
            If None, the splits will not be saved. 
            Default names of the splits are "train.csv", "dev.csv","test.csv"
        balanced_train_dev (bool, optional): If True, the train and test splits will be balanced. 
            Test will always remain umbalanced. Defaults to False.
        content_column (str, optional): Name of the column 
        labels_column (str, optional): Name of the column used for stratification when splitting.
            Remember that by default we are expecting datasets to have a 'labels' column.
            If `shuffle = True`, it will be used for stratification when necessary


    Returns:
        Dict[str, pd.DataFrame]: Dictionary whose keys are the split names and values are the split dataframes
    """

    # check values
    try:
        assert train + dev + test == 1.0
    except AssertionError:
        raise ValueError("Fractions for splitting must add up to `1.0`")

    # shuffle and stratify
    if shuffle:

        # count proportion of each label in the dataset
        class_weights = {}
        for label_value, sub_df in df.groupby(labels_column):
            class_weight = len(sub_df)/len(df)
            class_weights[str(label_value)] = class_weight
        # order the class weights
        class_weights = [v for _, v in class_weights.items()]

        # first we get the test split
        test_size = int(test * len(df))

        df_test = bibliome_sample_df_with_class_weights(
            df,
            target_column=labels_column,
            class_weights=class_weights,
            result_size=test_size
        )

        df_train_dev = bibliome_dataframe_difference(df, df_test)

        # now we get train and dev
        train_size = int(train * len(df))
        dev_size = int(dev * len(df))

        if not balanced_train_dev:
            # keep the same distribution as in the target column
            df_dev = bibliome_sample_df_with_class_weights(
                df_train_dev,
                target_column=labels_column,
                class_weights=class_weights,
                result_size=dev_size
            )
            df_train = bibliome_dataframe_difference(df_train_dev, df_dev)
            df_train = df_train.sample(frac=1, random_state=RANDOM_STATE)
        else:
            # uniform distribution
            class_weights = [1/len(class_weights) for _ in class_weights]

            df_dev = bibliome_sample_df_with_class_weights(
                df_train_dev,
                target_column=labels_column,
                class_weights=class_weights,
                result_size=dev_size
            )
            df_train = bibliome_dataframe_difference(df_train_dev, df_dev)

            # here we have to deal with unbalanced data
            # this will oversample or undersample depending on the class weights
            df_train = bibliome_sample_df_with_class_weights(
                df_train,
                target_column=labels_column,
                class_weights=class_weights,
                result_size=train_size,
                replace=True,  # for unbalanced data
            )

    else:
        # a very simple split
        df_train, df_dev, df_test = np.split(
            df,
            [int(train*len(df)), int((train + dev)*len(df))]
        )

    result_dict = {
        "train": df_train,
        "dev": df_dev,
        "test": df_test,
    }

    # save to a file
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for split_name, split_df in result_dict.items():
            split_df_path = os.path.join(save_dir, f"{split_name}.csv")
            split_df.to_csv(
                split_df_path,
                encoding="utf-8",
                lineterminator="\n",
                quoting=csv.QUOTE_NONNUMERIC,
            )

    return result_dict


def get_class_weights(
        df: pd.DataFrame,
        target_column: str,
) -> List:
    """Calculate the proportion of each label in the target column of the dataset
        Values of the target_column MUST be integers, starting from 0 till N

    Args:
        df (pd.DataFrame): dataframe whose target column is full of integers
        target_column (str): name of the target column

    Returns:
        List: list of proportions for each class.
    """

    # use a dict because we don't know in which orders the labels may appear on the dataset
    class_weights = {}
    # just count
    for label_value, sub_df in df.groupby(target_column):
        class_weight = len(sub_df)/len(df)
        class_weights[str(label_value)] = class_weight
    # order the class weights
    class_weights = [v for _, v in class_weights.items()]
    return class_weights


def bibliome_sample_df_with_class_weights(
        df: pd.DataFrame,
        target_column: str,
        class_weights: List,
        result_size: int,
        replace=False,
) -> pd.DataFrame:
    """Sample a dataframe while keeping the distribution of the target column

    Args:
        df (pd.DataFrame): _description_
        target_column (str): _description_
        class_weights (List): _description_
        result_size (int): size of output dataframe
        replace (bool): sample with replacement

    Returns:
        pd.DataFrame: _description_
    """

    # the idea is to sample from each category, and then join and shuffle
    samples_list = []
    for label_value, sub_df in df.groupby(target_column):
        # label values must be ints!
        sample_size = int(result_size*class_weights[label_value])
        sample_sub_df = sub_df.sample(
            sample_size,
            random_state=RANDOM_STATE,
            replace=replace,
        )
        samples_list.append(sample_sub_df)
        # join and shuffle
    result_df = pd.concat(samples_list).sample(
        frac=1, random_state=RANDOM_STATE)
    return result_df


def bibliome_dataframe_difference(
        df1, df2,
) -> pd.DataFrame:
    """Return a dataframe of all in df_1 and not in df_2
    https://stackoverflow.com/questions/48647534/find-difference-between-two-data-frames

    Args:
        df_1, df_2 : Dataframes
    Returns:
        pd.DataFrame: df_1 without values from df_2
    """
    df_3 = df1[
        ~df1.astype(str).apply(tuple, 1).isin(df2.astype(str).apply(tuple, 1))
    ]
    return df_3


def bibliome_build_DatasetDict(
        dataframe_dict: Dict[str, pd.DataFrame]
) -> DatasetDict:
    """Build a HuggingFace DatasetDict object in pytorch format from a dictionary of DataFrames.
    The original keys are keps. Each dataframe is converted into a Huggingface Dataset object.

    Args:
        dataframe_dict (Dict[str, pd.DataFrame]): 
            Dictionary of Dataframes.

    Returns:
        DatasetDict: DatasetDict object built from original dicionary
    """
    dataset_dict = DatasetDict({
        key: bibliome_build_Dataset(df)
        for key, df in dataframe_dict.items()
    })
    dataset_dict.set_format("torch")

    return dataset_dict


def bibliome_build_Dataset(
        dataframe: Union[pd.DataFrame, Dict]
) -> Dataset:
    """Build a HuggingFqce Dataset object in pytorch format from a dictionary of DataFrames.

    Args:
        dataframe (pd.DataFrame): 
            Dataframe to be converted into a pytorch Dataset.

    Returns:
        Dataset: Dataset object built from original dataframe
    """
    if isinstance(dataframe, dict):
        dataframe = pd.DataFrame(dataframe)
    dataset = Dataset.from_pandas(
        dataframe, preserve_index=False,
    )
    dataset.set_format("torch")
    return dataset


########################
# Utils for cleaning strings
########################
EMOJI_PATTERN = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

# these suffixes appear at the beginning or end of the strings
# and they provide no conent
AFFIXES_TO_BE_REMOVED = [
    'more resources\n\n\nFull Text Sources',
    'more resources\n\n\nFull Text Sources\n\n\nMiscellaneous',
    'Italia Olivicola Consorzio Nazionale',
    'more resources\n\n\nFull Text Sources\n\n\nResearch Materials',
    'Texas A&M AgriLife Research and Extension Center at Amarillo',
    'Journal of Pest Science',
    'Groenten & Fruit Actueel',
    'Egyptian Journal of Biological Pest Control',
    '20℃\n有效期：1 Year\n货期：1 Month\n其他：\n产品介绍：\n订货信息',
    'Texas Crop and Weather Report',
    'Beyond Pesticides Daily News Blog',
    'Pugliapress - Quotidiano online',
    'more resources\n\n\nFull Text Sources\n\n\nResearch Materials\n\n\nMiscellaneous',
    'Quotidiano di Notizie Gratuite',
    'European Journal of Plant Pathology',
    'Global Plant Protection News',
    'Oltre Free Press - Quotidiano di Notizie Gratuite',
    'Journal of Plant Pathology',
    'IVIA - Generalitat Valenciana',
    'International Journal of Tropical Insect Science',
    'The Document has moved here',
    'Magazine di informazione regionale della Puglia',
    'Información económica y empresarial de Huelva',
    'سامانه علمسنجی اعضای هیات علمی',
    'Roma, 18 set. (Adnkronos)',
    'reviewed Open Access journal.',
    'Environmental Science and Pollution Research',
    'Journal of Plant Diseases and Protection',
    'California Fresh Fruit Magazine',
    'Center for Invasive Species Prevention',
    'Video Dailymotion Watch fullscreen',
    'Roma, 24 ott. (Adnkronos)',
    'Italia Olivicola National Consortium',
    'Independent and short film streaming platform.',
    'Vidéo Dailymotion Watch fullscreen Font',
    'Crise de la filière viti',
    'Biblioteca Virtual da FAPESP',
    'Consiglio Regionale della Puglia',
    'DI SPACCIO DEL NORD',
    'Microbes protect a leaf beetle',
    'Fruit &amp; Vegetables News',
    'Free Stock Photos from Dreamstime',
    'Free & Royalty-Free Stock Photos from Dreamstime',
    'Israel Agricultural Technology & innovations Hub',
    'Fruit & Vegetable Magazine',
    'Idea Radio nel Mondo',
    'FI,30,WWF,125,xanthi asteras tripolis livestreaming,1,xanthi panathinaikos livestreaming,1,XHMEIA,1,Xylella fastidiosa,107,Xylotrechus chinensis,1,YARA,3,ZEA,2,ZIZANIOKTONA,3,ZOOTECHNIA,1,\n\n\n\nΣχόλια',
    '\d\d\d\d GoDaddy, LLC. All rights reserved. Privacy Policy',
    'Instituto de Agricultura Sostenible',
    'Instituto Valenciano de Investigaciones Agrarias (IVIA)',
    'Álvaro Guerrero\nSan Rafael, Ver.',
    '2018 中国地质图书馆版权所有 京ICP备05064691号 京公网安备11010802017129号\n地址：北京市海淀区学院路29号 邮编：100083\n电话：办公室：(+86 10)66554848；文献借阅、咨询服务、科技查新：66554700',
    'Library مفت الکترونیکی کتابتون',
    'الجمعية العربية لوقاية النبات',
    'la voce a Sud',
    'City of Buena Park',
    'Corriere di Puglia e Lucania',
    'time.news - Time News',
    'Centre for Functional Ecology',
    'City of Orange California',
    'National Agricultural Research Center',
    'Rotary Club Le Port',
    'Il periodico dei viticoltori italiani',
    'Millevigne - Il periodico dei viticoltori italiani',
    'Economic and business information on Huelva',
    'Performance of the domestic Bt',
    'BOLETIN OFICIAL REPUBLICA ARGENTINA',
    'In pubblicazione il bando della sottomisura 8.1',
    'Martínez de la Torre, Ver.',
    'O InnovPlantProtect (InPP) e a Direção',
    'Offre publiée il y a 3 jours',
    'CABI Agriculture and Bioscience',
    'Academic Journals - African Journal of Agricultural Research',
    'Studio Aperto Video | Mediaset Infinity',
    'The British Society for Plant Pathology',
    'Composts, Soils, Mulch, Ginseng',
    "Greg Alder's Yard Posts: Southern California food gardening",
    'Discover Open Access Resources',
    'KAMA International Organic pvt ltd.',
    'Ré à la Hune',
    'ScienceDaily - Verve times',
    'University of Florida, Institute of Food and Agricultural Sciences',
    'Eden Prairie Local News',
    'Annals of Forest Science',
    'Sabuj Sathi | Agriculture Solution',
    'La Voce di Maruggio',
    'vous ou créez un compte.',
    'Fonte immagine: Antenna SudSALENTO',
    'Fonte immagine: il Giornale di PugliaBARI',
    'Tous nos articles et événements',
    'Nieuws en kennis voor de akkerbouwers',
    'ALTIARA | Naturalmente productivo',
    'الموقع العلمي للدكتور الحسن اشباني',
    'Business24 La TV del Lavoro',
    "FR\nEN\nMétéo\nNewsletters\nMagazine\nSe connecter\nJe m'abonne pour 1€\nViticulture\nOenologie\nCommerce/Gestion\nPolitique\nGens du vin\nSERVICES\nNewsletters\nMétéo\nVindexer\nVidéos/Podcasts\nEmploi\nIndex des produits œnologiques\nMachines à vendanger\nViniconnect\nIntervignes\nAgenda\nCommunication\nRechercher\nS'abonner\nViticulture\nOenologie\nCommerce/Gestion\nPolitique\nGens du vin\nServices\n<",
]

# Most common hyphen-like characters in unicode
# such as the regular hyphen-minus (U+002D), figure dash (U+2012), en dash (U+2013), em dash (U+2014), and others.
HYPHEN_LIKE_CHARACTERS = re.compile(
    r'[\u002D\u058A\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]'
)
HYPHEN_LIKE_CHARACTERS_FLANKING = re.compile(
    r'^[\u002D\u058A\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]+|[\u002D\u058A\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]+$'
)
# Most common quotation-mark-like characters in unicode
QUOTATION_MARK_LIKE_CHARACTERS = re.compile(
    r'[\u00BB\u00AB\u25B7\u2018\u2019\u201C\u201D\u2039\u203A\u300C\u300D\u300E\u300F\u301D\u301E\u301F\uFE41\uFE42\uFE43\uFE44\uFF02\uFF07\uFF62\uFF63]'
)

# URL pattern
# URL_PATTERN = re.compile(r'\bhttps?://\S+\b', flags=re.IGNORECASE)
URL_PATTERN = re.compile(r'\b(?:https?://|www\.)\S+\b', flags=re.IGNORECASE)
# Date pattern
# Dates in formats YYYY-MM-DD, DD-MM-YYYY, DD.MM.YYYY, DD/MM/YYY, etc....
DATE_PATTERN = re.compile(
    r"\b\d{4}[.-/]\d{1,2}[.-/]\d{1,2}\b|\b\d{1,2}[.-/]\d{1,2}[.-/]\d{4}\b|\b\d{1,2}[.-/]\d{1,2}[.-/]\d{1,2}\b")

# Hours pattern
# e.g. 10:30 AM, 15:45, or 2023-05-15 10:30:00. Also, 15:45:30 and 2023-05-15T12:30:45+00:00
HOURS_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b", flags=re.IGNORECASE)


def clean_multilingual_string(input_string: str) -> str:
    if input_string is None:
        return None
    if not isinstance(input_string, str):
        # print(f"Not str! The content is of type {type(input_string)}")
        return input_string

    # clean at most 10 times:
    counter = 0

    previous_string = ''
    cleaned_string = input_string
    cleaned_string = HYPHEN_LIKE_CHARACTERS.sub('-', cleaned_string)

    # Remove HTML tags
    cleaned_string = re.sub('<[^>]+>', '', cleaned_string)

    # Remove URLs
    cleaned_string = URL_PATTERN.sub('', cleaned_string)

    # Remove dates
    cleaned_string = DATE_PATTERN.sub('', cleaned_string)

    # Remove hours
    cleaned_string = HOURS_PATTERN.sub('', cleaned_string)

    # remove irrelevant affixes
    for affix in AFFIXES_TO_BE_REMOVED:
        cleaned_string = cleaned_string.replace(affix, "")

    # normalize quotation marks
    cleaned_string = QUOTATION_MARK_LIKE_CHARACTERS.sub("'", cleaned_string)

    while cleaned_string != previous_string:
        previous_string = cleaned_string

        # remove flanking whitespaces and hyphens
        cleaned_string = re.sub(r'^\s+|\s+$', '', cleaned_string)
        cleaned_string = HYPHEN_LIKE_CHARACTERS_FLANKING.sub(
            '', cleaned_string
        )
        cleaned_string = cleaned_string.strip("-")

        # Remove extra white spaces
        cleaned_string = re.sub('\s+', ' ', cleaned_string).strip()

        # remove irrelevant suffixes
        # some suffixes are very short
        # delete suffixes with three or less tokens
        split_on_hyphen = cleaned_string.rsplit("-", 1)
        if len(split_on_hyphen) > 1:
            [left, right] = split_on_hyphen
            if right and len(right.strip().split()) <= 3:
                cleaned_string = left
        split_on_bar = cleaned_string.rsplit("|", 1)
        if len(split_on_bar) > 1:
            [left, right] = split_on_bar
            if right and len(right.strip().split()) <= 3:
                cleaned_string = left

        # Remove single and double quotation marks at the beginning and end of the string
        cleaned_string = re.sub(r'[\'"][\'"]+', "'", cleaned_string)
        cleaned_string = re.sub(r'^[\'"]|[\'"]$', '', cleaned_string)
        cleaned_string = cleaned_string.strip("'").strip('"')

        # remove orphan ' ": '
        cleaned_string = re.sub(r'[\'"]\s*:', '', cleaned_string)

        # Remove emojis
        cleaned_string = EMOJI_PATTERN.sub(r'', cleaned_string)

        # Remove characters not in any human alphabet
        cleaned_string = ''.join(
            c for c in cleaned_string if unicodedata.category(c)[0] != 'C'
        )

        # Remove punctuation marks at the beginning and end of the string
        cleaned_string = cleaned_string.strip(string.punctuation)

        # remove digits at the beginning and end of the string
        cleaned_string = re.sub(r"^\d+|\d+$", "", cleaned_string)

        # if the cleaned string is now empty, stop
        if not cleaned_string:
            return None

        # clean at most 10 times:
        counter += 1
        if counter == 10:
            return cleaned_string

    # forcing cleaning of quotation marks
    cleaned_string = cleaned_string.strip("'").strip('"')

    return cleaned_string
