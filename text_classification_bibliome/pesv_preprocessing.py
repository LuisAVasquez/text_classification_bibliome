"""Tools for preprocessing the PESV-VSI database"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re
from tqdm import tqdm
from pprint import pprint

import pandas as pd
from bs4 import BeautifulSoup
import nltk
from .bert_finetuning import bibliome_test_train_dev_split
from .bert_finetuning import bibliome_load_dataset_for_finetuning
from .preprocessing import get_class_weights

########################
# Utils for the raw dataset
########################


def bibliome_pesv_build_dataframe_from_columns(
    dataset_cols_dir: Union[str, os.PathLike],
    column_names: Union[str, List[str]]
) -> pd.DataFrame:
    """Build a dataframe from the column names of the PESV dataset

    Args:
        dataset_cols_dir (Union[str, os.PathLike]): 
            A directory containing one CSV file per each column of the VSI dataset.
            The VSI dataset is too big, so, it's been separated into one CSV file for each column
        column_names (Union[str, List[str]]): 
            Column names to be used

    Returns:
        pd.DataFrame: a Datafrane containing all the specified columns
    """

    # transform column_names to List[str]
    if isinstance(column_names, str):
        column_names = [column_names]

    # get all files from the directory
    columns_csvs = os.listdir(dataset_cols_dir)

    loaded_dataframes = []  # after loading, the dataframes will be merged

    for col_name in column_names:
        path_col_csv = f"{col_name}.csv"
        if path_col_csv not in columns_csvs:
            raise ValueError(
                f"Column '{col_name}' not found in {dataset_cols_dir}")

        path_col_csv = os.path.join(dataset_cols_dir, path_col_csv)
        col_df = pd.read_csv(
            path_col_csv,
            lineterminator="\n",
            # encoding="utf-8",
        )
        loaded_dataframes.append(col_df)

    # concatenate the dataframes

    joined_df = pd.concat(loaded_dataframes, axis=1)

    return joined_df


def process_subject_column(
    dataframe: pd.DataFrame,
    subject_column_name="sujet",
    has_subject_column_name="has_subject",
) -> pd.DataFrame:
    subject_col = dataframe[subject_column_name]

    has_subject_col = [
        1
        if (
            (isinstance(row, str) and row != 'None')
            or isinstance(row, int)
            # dealing with np.nan, which is a float
            or (isinstance(row, float) and row >= -99999)
        )
        else 0
        for row in subject_col
    ]
    dataframe[has_subject_column_name] = has_subject_col

    return dataframe


def get_content_type(
        target_column: str
) -> str:
    if "title" in target_column.lower():
        return "title"
    elif "abstract" in target_column.lower():
        return "abstract"
    elif "fulltext" in target_column.lower():
        return "fulltext"


def to_be_kept_simple(
    text_content: str
) -> bool:
    # ignore Nan
    if isinstance(text_content, float):
        return False
    # only keep strings
    if not isinstance(text_content, str):
        return False
    # ignore empty strings
    if not text_content:
        return False
    if text_content == 'None':
        return False
    return True


def to_be_kept(
    content_type: str = None,
    text_content: str = None,
    keep_titles=False,
) -> bool:
    """Detects scrapping errors.
    Returns False iff the text_content is detected as non-content from a website
    (i.e headers, menus, website names, etc)



    Args:
        content_type (str, optional): Whether the text is a title, abstract, or the full_text.
            Each content source uses somewhat different regex patterns.
            Defaults to None, but it is better to always provide it.
        text_content (str, optional): Text to be analyzed. Defaults to None.
        keep_titles (bool, optional): Whether to keep the name of the website, based on regex. Defaults to False. 
            That is, by default we discard the names of the websites.

    Raises:
        ValueError: When the content_type is not valid.
        content_type must be either 'title', 'abstract', or 'fulltext'

    Returns:
        bool: False iff our regex patterns detect the text is a scrapping error.
    """

    # drop nans
    if not to_be_kept_simple(text_content):
        return False

    # start to check if the scrapping failed
    text_content = text_content.strip()

    if is_dubious(text_content):
        return False

    if keep_titles:
        if is_title(text_content):
            return True
    else:
        if is_title(text_content):
            return False

    if content_type is None:
        content_type = get_content_type(content_type)

    if content_type == "title":
        return to_be_kept_title(text_content)
    elif content_type == "abstract":
        return to_be_kept_abstract(text_content)
    elif content_type == "fulltext":
        return to_be_kept_fulltext(text_content)
    else:
        raise ValueError(
            "content_type must be either 'title', 'abstract', or 'fulltext'")

    # by default, keep the content, just in case.
    return True

##################
# Dubious content :
# Not sure what happened while scrapping
# Maybe show it to the annotators
##################


def is_dubious(text_content: str) -> bool:
    """Say if the text content should be ignored for training
    because 
    - it may come from a failure of scrapping
    - it comes from a list of known things to ignore
    - it is too short

    Args:
        text_content (str): the textual content

    Returns:
        bool: True iff the content should be ignored for training
    """
    return (
        is_too_short(text_content)
        or is_known_dubious(text_content)
    )


def is_too_short(text_content: str) -> bool:
    """"""
    # Careful, some languages, like chinese, don't include spaces.
    # That's why one needs to check the string length:
    try:
        tokens = try_simply_tokenize(text_content)
        if len(tokens) < 2 and len(text_content) > 20:
            return False
        if len(tokens) <= 4:
            return True
        return False

    except:
        # None and nan values
        return True


def is_known_dubious(text_content: str) -> bool:
    return text_content in DUBIOUS_CONTENT


DUBIOUS_CONTENT = set(
    "X-MOL",

)


##################
# Regular expressions to avoid
##################

# regular expressions to avoid for title, abstract and full text

# Error messages

RE_error_messages = [
    # from experiments by Marie
    r"^JavaScript is not available\.$",
    r"^Error$",
    r"^Text$",
    r"^NA$",
    r"^Timeout error$",
    r"^post_title$",
    # from experiments by Luis
    r"^\d+$",
    r"^.*cookie.*$",
    r"^.*JavaScript.*$",
    r"^.*your browser.*$",
    r"^Please turn JavaScript on and reload the page$",
    r"^None$",
    r"^Loading(\.)*$",
    r"^Not Found$",
    r"^Access Restricted$",
    r"^Page Not Found$",
    r"^\[\]$",
    r"^blacklisted",
    r"^'NoneType' object has no attribute 'get'?$",
    r"^Copyright All Rights Reserved.+$"
    r"^PPlease Wait\.\.\.\s+\| Cloudflare$",
    r"^HTTPSConnectionPool.*",
    r"^HTTPConnectionPool.*",
    r"^Checking your browser.*",
    r"^Please update your browser$",
    r"^JavaScript n'est pas disponible\.$",
    r"^Vos données\. Votre expérience\.*$",
    r"^Before you continue to YouTube$",
    r"^Google Search Console$",
    r"^Just a moment(\.)*",
    r"^Your data\. Your experience\.$",
    r"^Access Denied$",
    r"^Need Help \?$",
    r"^Please Wait\.\.\. \| Cloudflare$",
    r"^Download PDF$",
    r"^\(нет голосов\) Loading\.\.\.$",
    r"^Article not found$",
    r"^Vos informations, votre expérience$",
    r"JavaScript 5",
    r"^Pardon Our Interruption$",
    r"^æ£å¨åå¾ï¼è¯·ç¨å...$",  # ??? Bad encoding ?
    r"^index$",
    r"^JavaScript",
    r"^We apologize for the inconvenience\.\.\.$",
    r"^Preparing your download$",
    r"^: 21 \(7\)$",
    r"^I tuoi dati, la tua esperienza$",
    r"^Verificaţi adresa de email$",
    r"^I tuoi dati. La tua esperienza.$",
    r"^Modification de page en cours$",
    r"^404$",
    r"^2022å¹´ä¸åå¹´æµ·å ³æªè·æ£ç«æ§æå®³çç©3.1ä¸ç§æ¬¡$",
    r"^Untitled$",
    r"^Sorry!$",
    r"^Thanks you$",
    r"^#25$",
    r"^æå¦ç§ç$",
]

RE_object_error_messages = re.compile(
    "|".join(RE_error_messages),
    flags=re.IGNORECASE
)


# Headers

RE_headers = [
    # from experiments by Luis
    r"^Home$",
    r"^Facebook$",
    r"^Idioma$",
    r"^Publications$",
    r"^Buscar$",
    r"^Search$",
    r"^Archives$",
    r"^Author Details$",
    r"^Facebook\s+Twitter\s+Instagram$",
    r"^Sequence Features$",
    r"^Discover open access scientific publications$",
    r"^Abbonati subito$",
    r"^News$",
    r"^Related Articles$",
    # translation : Frequenly Used Links
    r"^常用链接$",
    r"^Cultures$",
    r"^Browse Articles$",
    r"^Article Versions Notes$",
    r"^Datasets$",
    r"^Search Results$",
    r"^Browsing by Title$",
    r"^Search\. Read\. Cite\.$",
    r"^Agricultura$",
    r"^datosrevista$",
    r"^Journal$",
    r"^Table of Contents$",
    r"^Text Availability$",
    r"^Recherche$",
    r"^Agriculture$",
    r"^Noticias$",
    r"^Current Science$",
    r"^Sighting detail$",
    r"^Cloudflare$",
    r"^Mi cuenta\s+Cerrar Sesión$",
    r"^Ganadería Sostenible$",
    r"^Vidéos$",
    r"^.*pdf$",
    r"^Attualità$",
    r"^Actualités$",
    r"^Latest$",
    r"^Video$",
    r"^Ambiente$",
    # translation : Related products
    r"^Свързани продукти$",
    r"^Find Search Results$",
    # translation : Similar products
    r"^Podobné produkty$",
    r"^Articles$",
    r"^Notizie$",
    r"^Expert Groups$",
    r"^Categories$",
    r"^NOTÍCIAS$",
    r"^News Flash$",
    r"^Résultats de la recherche$",
    r"^References$",
    r"^Index$",
    r"^Search results$",
    r"^Advance articles$",
    r"^Pesquisa$",
    r"^Notícias$",
    r"^Browsing by Subject$",
    r"^Information$",
    r"^Home 1$",
    r"^Programs and Projects$",
    r"^Institucional$",
    r"^Collections$",
    r"^Tags$",
    # translation : Latest News
    r"^最新消息$",
    r"^Author Search Results$",
    # translation : Search Results
    r"^检索结果$",
    r"^Connexion$",
    r"^Derniers Articles$",
    r"^Notizie - Attualità$",
    r"^Notizie dal Comune$",
    r"^Data Reports$",
    r"^Single Sign-Off$",
    r"^TRIPS & CLUBS$",
    # translation : Document Search
    r"^文献搜索$",
    r"^Publication Info.$",
    # translation : Basic Article Information
    r"^文章基本信息$",
    r"^Fiestas$",
    r"^ART$",
    r"^Referencias$",
    r"^Journal$",
    # translation : browse by author
    r"^Sfoglia per Autore$",
    r"^Economía$",
    r"^Blogs?$",
    r"^b|Browse$",
    r"^Perfil do autor$",
    # translation : Reader Cloud Portal
    r"^Accueil$",
    r"^TikTok$",
    r"^MSN$",
    r"^Animaux$",
    r"^Société$",
    r"^Shop$",
    r"^Destaques$",
    r"^Teses de Doutorado$",
    # translation : similar products
    r"^Srodni proizvodi$",
    r"^Agricoltura$",
    r"^Avant d'accéder à Google$",
    r"^Subscription$",
    r"^Detalles de autor/a$",
    r"^Inicio$",
    r"^general profile$",
    r"^Make a Submission$",
    # translation : Suggestions
    r"^Savjetuje$",
    r"^Select your language$",
    # translation : similar products
    r"^Susiję produktai$",
    # translation : related products
    r"^Príbuzné produkty$",
    r"^Author: admin$",
    r"^Todas as Notícias$",
    r"^S|shop$",
    r"^Archive$",
    r"^Your search returned . results\.$",
    r"^Category: Uncategorized$",
    # translation : Search Results
    r"^検索結果$",
    # translation : latest news
    r"^Сўнгги янгиликлар$",
    r"^Research articles$",
    r"^Catalogo$",
    r"^Main navigation$",
    r"^Articles in 20..$",
    r"^Listar$",
    # translation : Personal user login
    r"^个人用户登录$",
    # translation : Related products
    r"^相关产品$",
    r"^Newsroom$",
    r"^Narrow Search$",
    r"^Limites de sujet actuelles :$",
    r"^Activities: Projects$",
    r"^acesso aberto$",
    # translation : Market prices and exchanges
    r"^Цени на тържища и борси$",
    r"^Welcome$",
    r"^Link Diretti$",
    r"^Búsqueda$",
    r"^Login • Instagram$",
    r"^Bloomberg$",
    r"^Calculating Ephemerides$",
    r"^Categoría$",
    r"^Search Papers$",
    r"^Economia$",
    r"^Notizie e Avvisi$",
    r"^Products$",
    r"^Main Menu$",
    r"^Main menu$",
    r"^Produtos Relacionados$",
    r"^All News$",
    r"^Artigo$",
    r"^Browsing Tesis by Title$",
    r"^Sobre la biblioteca$",
    r"^Thanks you$",
    r"^détails \| FAO$",
    r"^Presse$",
    r"^Search speeches$",
    r"^Banco de Notícias$",
    r"^Novedades$",
    r"^Homepage$",
    r"^Glosario$",
    r"^Program$",
    r"^Events$",
    r"^Contact Us$",
    r"^Politica$",
    r"^Discover$",
    r"^Publication$",
    r"^Search all articles$",
    r"^Documentos$",
    r"^seguici su$",
    r"^indexadores$",
    r"^Details$",
    r"^Información$",
    r"^Publications and data$",
    r"^Titulares$",
    r"^GET TO KNOW US$",
    r"^vos critères$",
    r"^Dictionnaire$",
    r"^Contacto$",
    r"^¡Suscríbete!$",
    r"^Lines of research$",
    r"^Metabuscador$",
    r"^Portale dei servizi$",
    # translation : News
    r"^Nieuws$",
    r"^Repositorio Digital$",
    r"^Pubblicazioni$",
    r"^Comunicati$",
    r"^Toute l'actualité$",
    r"^Aide à l'outil informatique$",
    r"^vikaspedia Domains$",
    r"^Job Posting$",
    r"^Shop$",
    r"^Shop$",
    r"^Shop$",
    r"^Shop$",
    r"^Shop$",
    r"^Shop$",
]

RE_object_headers = re.compile(
    "|".join(RE_headers),
    flags=re.IGNORECASE
)

# always avoid error messages and headers

RE_always_avoid = RE_error_messages + RE_headers

RE_object_always_avoid = re.compile(
    "|".join(RE_always_avoid),
    flags=re.IGNORECASE
)


# regular expressions to avoid for the title
RE_avoid_for_title = RE_always_avoid + [
    # from experiments by Marie
    r"^About$",
    r"^browser compatibility check",
    ###
    # from experiments by Luis
    ###
    r"^Please turn JavaScript on and reload the page\.",

    r"^ORCID$",
    r"^SeedQuest$",
    r"^X-MOL$",
    r"^Careers at UF$",
    r"^Browsing NTU Scholars$",
    r"^Moteur de recherche VSI$",
    # translation : Uzbek, List of citizes Registered for Affordable Housing
    r"^Арзон уй-жойлар ажратиш бўйича рўйхатга олинган фуқаролар рўйхати$",
    r"^Species belonging to selected taxon:$",
    r"^Card ar$",
    r"^Cb Phone Number$",
    # translation : Kanglong Biotechnology Official Online Shop
    r"^康朗生物官方商城$",
    # translation : Sf9 Huapi
    r"^Sf9 畫皮$",
    r"^University, Subject, Country, Region, World$",
    r"^Save BIG with \$9.99 \.COMs from GoDaddy!$",
    r"^Deze website maakt gebruik van cookies$",
    r"^Media Statements$",
    r"^Order Article Reprints$",
    r"^Bienvenue à Jonquerettes$",
    r"^Assistant inspecteur \(H/F\)$",
    r"^Faw IPM APK Download for Android$",
    r"^Capacitación$",
    r"^Acuerdos Ministeriales$",
    r"^Event$",
    r"^Feedback$",

    r"^Journal$",
    r"^Journal$",
]

RE_object_for_title = re.compile(
    '|'.join(RE_avoid_for_title),
    flags=re.IGNORECASE
)

# regular expressions to avoid for the abstract

RE_avoid_for_abstract = RE_always_avoid + [
    # from experiments by Marie
    r'^Access denied',
    r'^JavaScript is not available\.',
    r'^NA$',
    r'^post_title',
    r'^Please enable JavaScript',
    r'^Request Access',
    r'^We have detected invalid data in your input',
    r'^Sorry, for some reason',
    r'^Request unsuccessful',
    r'^All articles published by MDPI are made immediately available',
    r'^uses cookies',
    r'^utiliza Cookies',
    r'^Matéria não encontrada',
    r'^Something went wrong',
    r'^Wait a moment and try again',
    # from experiments by Luis
    r"^Faw IPM APK Download for Android",


]

RE_object_for_abstract = re.compile(
    '|'.join(RE_avoid_for_title),
    re.IGNORECASE
)

# regular expressions to avoid for the full text

RE_avoid_for_fulltext = RE_always_avoid + [
    # from experiments by Marie
    r'^Access denied',
    r'^JavaScript is not available.',
    r'^NA$',
    r'^post_title',
    r'^Please enable JavaScript',
    r'^Request Access',
    r'^We have detected invalid data in your input',
    r'^Sorry, for some reason',
    r'^Request unsuccessful',
    r'^All articles published by MDPI are made immediately available',
    r'^uses cookies',
    r'^utiliza Cookies',
    r'^Matéria não encontrada',
    r'^Something went wrong',
    r'^Wait a moment and try again',
    r'^privacy gate$',
    r'^\nPrivacy\n',
    r'^Please change your browser',
    # from experiments by Luis
    r"^ANALISI\s+Atlantide\s+Mezzaluna",
    r"^Click here to sign up",
    r"^Browsing Docencia by Title$",

]
RE_object_for_fulltext = re.compile(
    '|'.join(RE_avoid_for_title),
    flags=re.IGNORECASE
)


# Regular expressions which are names of websites and journals
RE_names = [
    ##
    # these look like journal or website names
    ##
    r"^Portal TecnoAgrÃcola$",
    r"^Portal Embrapa$",
    r"^puglialive.net$",
    r"^SINTA$",
    r"^SciELO Argentina - www\.scielo\.org\.ar$",
    r"^Argentina\.gob\.ar$",
    # translation : China Education Publishing & Media Group Co., Ltd.
    r"^中国教育图书进出口有限公司$",
    # translation : China National Publications Import & Export (Group) Corporation
    r"^中国图书进出口\(集团\)总公司$",
    # translation : Agricultural Science Institutions Knowledge Base Alliance
    r"^农科机构知识库联盟$",
    # translation : Home storage retrieval
    r"^首页仓储检索$",
    r"^Pakistan Journal of Zoology$",
    r"^Sighting detail - www\.ornitho\.it$",
    r"^PubAg$",
    r"^Agroscope$",
    r"^Journal of Plant Protection Research$",
    r"^Krushikendra\.com$",
    r"^Loading · OA.mg$",
    r"^Regione \| Canale 58 - la TV del territorio$",
    r"^AD$",
    r"^CIMMYT Publications Repository$",
    r"^EPPO Global Database$",
    r"^Create a SciFeed alert for new publications$",
    r"^Bibliothèque numérique DUDDAL.+$",
    r"^MediSys$",
    r"^SciELO Paraguay - scielo\.iics\.una\.py$",
    r"^Universidad Mayor de San Andrés$",
    r"^FAO\.org$",
    r"^Bibliothéque FST de Fès$",
    r"^PMI-Reboot$",
    r"^nordbayern\.de$",
    r"^Instituto Valenciano de Investigaciones Agrarias \(IVIA\) .*$",
    r"^Institut Valencià d'Investigacions Agràries \(IVIA\) - IVIA - Generalitat Valenciana$",
    r"^Generalitat Valenciana$",
    r"^Buonasera \| News$",
    r"^Eingeschleppter Schädling...Japankäfer erreicht Baden-Württemberg$",
    r"^CORC$",
    r"^16 e  législature$",
    r"^Portal da Agencia o Globo$",
    r"^DSpace Angular ::Estadísticas$",
    r"^Vidéos | Corse Net Infos$",
    r"^UniProt website fallback message$",
    # translation : Institutional Knowledge Repository of Chinese Academy of Tropical Agricultural Sciences
    r"^中国热带农业科学院机构知识库$",
    r"© Mairie de Maussane",
    r"^Browsing MT$",
    r"Menu\s+Home\s+Expert\s+search\s+Contact\s+About\s+AGUIA",
    r"^Digital Newspaper & Magazine Subscriptions$",
    r"^Consultas fitossanitárias e Receituário Agronômico$",
    r"^$",
    r"^@tube$",
    # translation : Search Results - Vip Journal Chinese Journal Service Platform
    r"^检索结果-维普期刊 中文期刊服务平台$",
    r"^WebPathology$",
    # translation : RISSH Search
    r"^RISS 검색$",
    r"^DPG Media Privacy Gate$",
    # translation : National Academy for Educational Research Bilingual Lexicon, Terminology, and Dictionary Information Website
    r"^國家教育研究院雙語詞彙、學術名詞暨辭書資訊網$",
    r"^gobertpartners\.com$",
    r"^International Institute of Tropical Agriculture \(IITA\)$",
    # translation : Browse NTU Scholars
    r"^瀏覽 NTU Scholars$",
    r"Provincia di Asti",
    r"^Asaja Asociación Agraria de Jóvenes Agricultores$",
    r"^View Gos Unik$",
    r"^The Plant Pathology Journal$",
    r"^Universitas Jenderal Soedirman \(UNSOED\) Purwokerto \[svr1\]$",
    r"^Profil Dosen$",
    r"^Les services de l'État dans le Gard$",
    r"^Sun Sentinel$",
    # translation : Baidu scholar
    r"^百度学术$",
    r"^Comune di Gerenzano Provincia di Varese$",
    r"^ACIAR Australia$",
    r"^AgriSolución$",
    r"^Università del Salento$",
    r"^Browsing UT$",
    r"^Hoвости | Turkmen.News$",
    r"^North American Plant Protection Organization$",
    r"^Universitas Jenderal Soedirman \(UNSOED\) Purwokerto \[svr3\]$",
    r"^Invasive Species Compendium (ISC) | CABI$",
    # translation : Reader Cloud Portal
    r"^重组蛋白$",
    # translation :  Kanglong Biotechnology Official Online Shop
    r"^最新消息$",
    r"^Canale 7$",
    r"^Publication Search \| US Forest Service Research and Development$",
    r"^idw – Informationsdienst Wissenschaft$",
    r"^ScienceDirect Topics in Agricultural and Biological Sciences$",
    r"^Agro Excelencia$",
    r"^Frontiers in Physiology \| Articles$",
    r"^Home - Akse Media$",
    # translation : Administration of the Federal Service for Veterinary and Phytosanitary Surveillance for Tyumen Oblast, Yamalo-Nenets Autonomous Okrug, and Khanty-Mansi Autonomous Okrug
    r"^Управление Федеральной службы по ветеринарному и фитосанитарному надзору по Тюменской области,Ямало-Ненецкому и Ханты-Мансийскому автономным округам$",
    # translation : Agricultural Institutions Knowledge Base Alliance
    r"^农业机构知识库联盟$",
    r"^SICRIS$",
    r"^Monde scientifique$",
    r"^Optica Publishing Group$",
    r"^Comune di Castellana Grotte$",
    r"^guce$",
    r"^Artificial Intelligence in Cancer$",
    r"^The Healthy Journal - Gluten, Dairy, Sugar Free Recipes, Interviews and Health Articles$",
    r"^ScienceDirect\.com \| Science, health and medical journals, full text articles and books\.",
    r"^Biological Forum-An International Journal$",
    r"^Le Journal Catalan$",
    r"^agroorganico$",
    r"^Pakistan Journal of Agricultural Research$",
    # translation : aTkati Agricultural and Fishery Product Export Support Information
    r"^aTkati 농수산식품수출지원정보$",
    r"^Zinfos974$",
    r"^Sf9 畫皮$",
    r"^Cronaca$",
    r"^• Provincia di Asti$",
    r"^MINISTERO DELLE POLITICHE AGRICOLE ALIMENTARI E FORESTALI$",
    r"^The Gabber$",
    r"^Sigpro$",
    r"^Comune di Omegna$",
    r"^Mazii - Rated #1 Japanese English Dictionary Online$",
    r"^Maryland Biodiversity - County Records$",
    r"^SIGAA$",
    r"^Region Vrbas$",
    r"^Arboricultural Association$",
    r"^CIHEAM Bari$",
    r"^Frontiers in Insect Science \| Invasive Insect Species$",
    r"^GOVPH$",
    r"^Informatics Publishing Limited$",
    # translation : Agricultural Machinery News (Co., Ltd.)
    r"^농기자재신문\(주\)$",
    # translation : Directorate of Plant Food Protection
    r"^Direktorat Perlindungan Tanaman Pangan$",
    # translation : Article Search - Northwest Agriculture and Forestry Journal Network
    r"^文章搜索——西北农林期刊网$",
    r"^Lic\. Alejandra Daguerre$",
    r"^Browsing Escuela de Ingeniería Agropecuaria by Title$",
    r"^Markets and Trade – Food and Agriculture Organization of the United Nations$",
    r"^Browse \| ASHS$",
    r"^zmianynaziemi\.pl$",
    r"^Revista Torreón Universitario$",
    r"^Actualités de la VSI$",
    r"^UC Statewide IPM Program \(UC IPM\)$",
    r"^Document card \| FAO$",
    r"^Frontiers in Insect Science$",
    r"^AMiner$",
    r"^Frontiers in Plant Science \| Articles$",
    r"^Search \| GOV.SI$",
    r"^doiSerbia$",
    r"^Regione autonoma Valle d'Aosta$",
    r"^International Plant Protection Convention$",
    r"^Novice - Občina Ljutomer$",
    r"^Biblioteca Digital INIA$",
    r"^India Biodiversity Portal$",
    r"^Podoslonami.pl$",
    r"^Universidad Politécnica de Madrid$",
    r"^Green Blog$",
    r"^RPubs$",
    r"^BioProject$",
    r"^InterPro$",
    r"^Animal Diversity Web$",
    r"^Observatorio de I\+D\+i UPM$",
    r"^Monochamus spp.$",
    r"^INIAV, IP$",
    r"^Comune di Verbania$",
    r"^Le blog d'Albert Amgar$",
    r"^huanglongbing$",
    r"^AgrocienciaÂ vol.46Â nÃºmero4; S1405-31952012000400007$",
    r"^Ossola 24$",
    r"^ARIF PUGLIA$",
    r"^Comté de Provence$",
    r"^Flashnews.gr$",
    r"^Dra. Myriam Elías Santos$",
    r"^Comune di Valmacca$",
    r"^Prodotti per l'Agricoltura$",
    r"^CCSE$",
    r"^Disciplinare.it$",
    r"^UDSspace$",
    r"^Camera dei deputati$",
    r"^- FruitJournal$",
    r"^InfoAgronomo$",
    r"^UnB Pesquisa$",
    r"^Gente en Valencia$",
    r"^Noticias Jurídicas$",
    r"^Invaio X Gli Ulivi$",
    r"^Products | Syngenta$",
    r"^Association Bernard Gregory$",
    r"^agronoticias.es$",
    r"^AgroNotizie - Notizie agricoltura$",
    r"^Regione Lazio$",
    r"^University of Zimbabwe Libraries$",
    r"^Alice: Search$",
    r"^AgroAvances$",
    r"^Jurnal Cyber Tech$",
    r"^La Rioja | Rioja2.com$",
    r"^BIBLIOS$",
    r"^UMR PVBMT$",
    r"^Epernay$",
    r"^#tobrfv - Explore$",
    r"^Publication card | FAO$",
    r"^InforCNA$",
    r"^Ansamed$",
    r"^nutrixrevolution$",
    r"^Osservatorio Agromafie$",
    r"^Citricos.com$",
    r"^MycoKeys$",
    r"^Alerta Diário$",
    r"^Avvisi$",
    r"^ProfessorJohnMumford$",
    r"^RCI Guadeloupe$",
    r"^Les actualités communales$",
    r"^BJA Ferramenta de Leitura$",
    r"^fhalmería$",
    r"^fhalmería$",
]

RE_object_names = re.compile(
    '|'.join(RE_names),
    flags=re.IGNORECASE
)


def to_be_kept_title(
    text_content: str
) -> bool:
    if re.match(RE_object_for_title, text_content) is not None:
        # there was a match with the patterns to avoid
        # Do not keep the text!
        return False

    else:
        return True


def to_be_kept_abstract(
    text_content: str
) -> bool:
    if re.match(RE_object_for_abstract, text_content) is not None:
        return False
    else:
        return True


def to_be_kept_fulltext(
    text_content: str
) -> bool:
    if re.match(RE_object_for_fulltext, text_content) is not None:
        return False
    else:
        return True


def is_title(
        text_content: str
) -> bool:
    if re.match(RE_object_names, text_content):
        return True
    else:
        return False


def is_error_message(
        text_content: str
) -> bool:
    """True if the string corresponds to an error message"""
    # filter out nans and empty strings
    if not to_be_kept_simple(text_content):
        return True

    if re.match(RE_object_error_messages, text_content) is not None:
        return True
    else:
        return False


########################
# Utils for parsing XML
########################

def extract_trafilatura_fulltext_abstract_title(
        xml_string: str
) -> List[str]:

    parsed_title = None
    parsed_abstract = None
    parsed_fulltext = None

    if not to_be_kept_simple:
        return (None, None, None)
    if not isinstance(xml_string, str):
        return (None, None, None)
    soup = BeautifulSoup(xml_string, "lxml")

    try:
        parsed_title = soup.find("title").get_text().strip()
        if not to_be_kept_simple(parsed_title):
            parsed_title = None
    except:
        pass
    try:
        parsed_abstract = soup.find("abstract").get_text().strip()
        if not to_be_kept_simple(parsed_abstract):
            parsed_abstract = None
    except:
        pass
    try:
        parsed_fulltext = soup.find("text").get_text().strip()
        if not to_be_kept_simple(parsed_fulltext):
            parsed_fulltext = None
    except:
        pass

    return (parsed_title, parsed_abstract, parsed_fulltext)


def add_columns_parsed_from_xml(
        df: pd.DataFrame,  # must contain column "trafilatura_extracted_xml_tei"
        columns=None
) -> pd.DataFrame:

    if columns is None:
        columns = [
            'parsed_trafilatura_title',
            'parsed_trafilatura_abstract',
            'parsed_trafilatura_fulltext',
        ]

    # initialize a dictionary to store the parsed results
    parsed_lists = {}
    for col in columns:
        parsed_lists[col] = []

    for html_string in tqdm(df['trafilatura_extracted_xml_tei']):
        (parsed_title, parsed_abstract, parsed_fulltext) = extract_trafilatura_fulltext_abstract_title(
            html_string
        )

        for col in columns:
            if "title" in col:
                parsed_lists[col].append(parsed_title)
            if "abstract" in col:
                parsed_lists[col].append(parsed_abstract)
            if "fulltext" in col:
                parsed_lists[col].append(parsed_fulltext)

    for col in columns:
        df[col] = parsed_lists[col]

    return df

########################
# Utils for cleaning the dataset
########################


def get_clean_dataset_add_has_subject_column(
    dataset: pd.DataFrame,
    target_column: str,
    keep_titles: bool = True,
) -> pd.DataFrame:

    content_dataframe = dataset[[target_column, 'sujet']]
    content_type = get_content_type(target_column)

    # A new dataframe to delete duplicates and detect subjects
    new_dataframe = {
        target_column: [],
        "has_subject": []
    }

    for text_content, sub_df in content_dataframe.groupby(
        by=target_column,
        dropna=False,
    ):
        # test if the content will be kept
        if to_be_kept(
            content_type=content_type,
            text_content=text_content,
            keep_titles=keep_titles
        ):
            # check if has subject
            aux_sub_df = process_subject_column(
                sub_df
            )
            has_subject = int(0)
            if aux_sub_df['has_subject'].sum() > 0:
                # got a subject at least one time
                has_subject = int(1)

            new_dataframe[target_column].append(text_content)
            new_dataframe['has_subject'].append(has_subject)

    new_dataframe = pd.DataFrame(new_dataframe)
    return new_dataframe

########################
# Utils for doing basic statistics
########################


def try_simply_tokenize(
        text
):
    try:
        # because our strings are multilingual, we cannot use a single tokenizer for them
        return text.split()
    except:
        return []


def basic_stats(df: pd.DataFrame, target_column: str = None):

    summary_result = {}

    # pprint(df.describe())
    print()
    pprint(df.info())
    print()
    value_counts = df['has_subject'].value_counts()
    pprint(value_counts)
    value_counts.plot.bar()
    summary_result["value_counts"] = value_counts.to_dict()

    if target_column is not None:
        summary_result["target_column"] = target_column
        # study text length
        text_length_study = (
            df[target_column].apply(try_simply_tokenize).apply(len).describe()
        )
        pprint(text_length_study)
        summary_result["text_length_study"] = text_length_study.to_dict()

    return summary_result


def basic_overview(
        df: pd.DataFrame,
        target_column: str,
        keep_titles: bool = True,
) -> None:
    analyzed_dataframe = get_clean_dataset_add_has_subject_column(
        df,
        target_column=target_column,
        keep_titles=True
    )
    summary = basic_stats(analyzed_dataframe, target_column)

    return summary


########################
# Utils for extracting relevant sentences
########################
try:
    nltk.download("punkt")
    nltk.download("stopwords")
except:
    pass

# Keywords and keyphrases used to search for documents
# String case is ignored
RE_for_relevant_sentences = [
    # Fusarium oxysporum f. sp. cubense Tropical race 4
    # q=allintext:+fusarium+oxysporum+tropical
    "fusarium",
    "oxysporum",
    "tropical",
    # Xylella fastidiosa
    # q=allintext:+xylella
    "xylella",
    # Bursaphelenchus xylophilus
    # q=allintext:+Bursaphelenchus+xylophilus
    "Bursaphelenchus",
    "xylophilus",
    # Bactrocera dorsalis
    # q=allintext:+Bactrocera+dorsalis
    "Bactrocera",
    "dorsalis",
    # Candidatus Liberibacter spp.
    # q=allintext:+huanglongbing
    "huanglongbing",
    # Popillia japonica
    # q=allintext:+Popillia+japonica
    "Popillia",
    "japonica",
    # Dépérissement de la vigne
    # q=allintext:flavescence
    "flavescence",
    # Généralités
    # q=allintext%3A%22first+report+plant+disease%22+OR+%22new+plant+health%22
    "first report plant disease",
    "new plant health",
    # ToBRFV
    # q=allintext:+ToBRFV
    "ToBRFV",
    # Spodoptera frugiperda
    # q=allintext:+spodoptera+frugiperda
    "spodoptera",
    "frugiperda",
    # Bretziella fagacearum
    # q=allintext:+oak+wilt+Bretziella
    "oak",
    "wilt",
    "Bretziella",
    # Agrilus planipennis
    # q=allintext:+Agrilus+planipennis
    "Agrilus",
    "planipennis",
    # Thaumatotibia leucotreta
    # q=allintext:+Thaumatotibia+leucotreta
    "Thaumatotibia",
    "leucotreta",
    # Xylotrechus chinensis
    # q=allintext:+Xylotrechus+chinensis
    "Xylotrechus",
    "chinensis",
    # Toumeyella parvicornis
    # q=allintext:+Toumeyella+parvicornis
    "Toumeyella",
    "parvicornis",
    # Ceratocystis platani
    # q=allintext:+Ceratocystis+platani
    "Ceratocystis",
    "platani",
    # Ceratocystis platani
    # q=allintext:+chancre+du+platane
    "chancre",
    "platane",
]

# Case does not matter:
# The relevant terms may be on the beginning of the content
RE_object_for_relevant_sentences = re.compile(
    '|'.join(RE_for_relevant_sentences),
    flags=re.IGNORECASE
)


def contains_keywords(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    if re.search(RE_object_for_relevant_sentences, sentence):
        return True
    else:
        return False


def try_extract_relevant_sentences(
    content: str
) -> str:

    try:
        # keep only those sentences containing at least one keyword/keyphrase

        # split in paragraphs
        paragraphs = [p.strip() for p in content.split("\n") if p]

        all_paragraph_sentences = [
            nltk.tokenize.sent_tokenize(paragraph) for paragraph in paragraphs
        ]
        # flatten the list
        all_sentences = [
            sent
            for sentence_list in all_paragraph_sentences
            for sent in sentence_list
        ]
        all_sentences = [sent.strip() for sent in all_sentences]

        # filter for sentences containing keywords
        relevant_sentences = [
            sent for sent in all_sentences
            if contains_keywords(sent)
        ]
        # when no keywords are in the content, return the content
        if not len(relevant_sentences):
            return None

        relevant_sentences = ".".join(relevant_sentences)
        return relevant_sentences

    except:
        return None


def add_relevant_sentence_columns(
        df: pd.DataFrame,
        target_columns: Union[str, List[str]] = None,
        keep_content=True,
) -> pd.DataFrame:

    if target_columns is None:
        target_columns = [
            "parsed_trafilatura_abstract",
            "parsed_trafilatura_fulltext",
        ]
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # initialize a dictionary to store the parsed results
    results_lists = {}
    for col in target_columns:
        results_lists[col] = []

    for col in target_columns:
        for text_content in tqdm(
            df[col]
        ):
            relevant_sentences = try_extract_relevant_sentences(
                text_content
            )
            if relevant_sentences is None:
                if keep_content:
                    new_entry = text_content
                else:
                    new_entry = None
            else:
                new_entry = relevant_sentences

            results_lists[col].append(new_entry)

    for col in target_columns:
        if keep_content:
            suffix = "keep_original_content"
        else:
            suffix = "only_relevant_sentences"
        df[f"sentence_with_keywords_{col}_{suffix}"] = results_lists[col]

    return df


#######
# Utilities for splitting a cleaned and filtered dataset
#######


def split_and_save_dataset(
        input_path: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        balanced_train_dev: bool = False,
        labels_column_name: str = "has_subject"
):
    """Split a dataframe into train, test, and dev. Save the splits to a directory.
    We assume that the file name is the same as the name of the column containing the content.
    e.g. 'trafilatura_title.csv' should have the text content in the 'trafilatura_title' column


    Args:
        input_path (Union[str, os.PathLike]): Path to CSV with the dataframe
        output_path (Union[str, os.PathLike]): Path to directory for saving the splits
        balanced_train_dev (bool, optional): Whether or not to balance the train and dev splits
        labels_column_name (str, optional) : name of the column containing the labels for classification
    """

    full_df_path = input_path
    text_column_name = os.path.basename(full_df_path).removesuffix(".csv")

    print(f"Now splitting: {text_column_name}")

    # load the dataframe
    df = bibliome_load_dataset_for_finetuning(
        full_df_path,
        text_column_name=text_column_name,
        labels_column_name=labels_column_name,
    )
    # rename the columns to their original names
    df.columns = [text_column_name, labels_column_name]

    # make the split
    is_balanced = (
        "balanced_dev_train" if balanced_train_dev else "unbalanced")
    save_dir = os.path.join(output_dir, is_balanced, text_column_name)

    df_split = bibliome_test_train_dev_split(
        df=df,
        train=0.8,
        test=0.1,
        dev=0.1,
        shuffle=True,  # we must shuffle
        save_dir=save_dir,
        content_column=text_column_name,
        labels_column=labels_column_name,
        balanced_train_dev=balanced_train_dev,
    )

    print("-"*10)

    return df_split
