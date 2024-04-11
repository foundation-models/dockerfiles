import json
import logging
import os
import re
from collections.abc import Mapping
from functools import cache
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

import requests
import spacy
import torch
import yaml
from hydra.utils import instantiate
from numpy import nan
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from pandas import (DataFrame, concat, json_normalize, read_csv, read_fwf,
                    read_parquet)
from tqdm import tqdm

log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

try:
    import fasttext
except ImportError:
    log.warning("fasttext not installed, please install it if you use fasttext")
try:
    from pdfreader import SimplePDFViewer
except ImportError:
    log.warning("pdfreader not installed, please install it if read PDF files")
    pass

try:
    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          utility)
except:
    log.warning("pymilvus not installed, please install it if you use Milvus")
    pass
try:
    import trafilatura  # for web crawling
except ImportError:
    log.warning(
        "trafilatura not installed, please install it if you use Web crawling")
    pass
try:
    import fasttext
    from fastlangid.langid import LID
except (ImportError, ModuleNotFoundError):
    log.warning(
        "fasttext and fastlangid not installed, please install it if you use language detection")
    pass



class RegexNER():
    def __init__(self, conf):
        self.email_regex = re.compile(conf["email"])
        self.web_regex = re.compile(conf["web"])
        self.linkedin_regex = re.compile(conf["linkedin"])
        self.replace_dict = {'w:': '', 'W:': '', 'blog:': '', "Blog:": ''}

    def replace_all(self, x):
        for k, v in self.replace_dict.items():
            x = x.replace(k, v)
        return x

    def get_entities(self, text, df=DataFrame()):
        """
        extract the named entities by regex.
        """
        emails = self.email_regex.findall(text)
        webs = [w.group(0) for w in self.web_regex.finditer(text)]
        linkedin = [l.group(0) for l in self.linkedin_regex.finditer(text)]
        linkedin = [
            l for l in linkedin if 'linkedin' in l and l != 'linkedin.com']
        linkedin = [l for l in linkedin if not any(
            [c in l for c in ['@', 'malito']])]
        linkedin = [l[:-1] if l[-1] == '>' else l for l in linkedin]
        webs = [w for w in webs if ((w not in linkedin) and (w not in emails))]
        webs = [w for w in webs if not any([c in w for c in ['@', 'malito']])]
        webs = [self.replace_all((w if w[-1] != '.' else w[:-1]))
                for w in webs]

        df = concat([df, DataFrame(
            [{'start': 0, 'end': 0, 'entity': 'EMAIL', 'name': item} for item in emails])], ignore_index=True)
        df = concat([df, DataFrame(
            [{'start': 0, 'end': 0, 'entity': 'LINKEDIN', 'name': item} for item in linkedin])], ignore_index=True)
        df = concat([df, DataFrame(
            [{'start': 0, 'end': 0, 'entity': 'URL', 'name': item} for item in webs])], ignore_index=True)

        """
        aggregation_functions = {'start': 'sum', 'end': 'sum', 'entity': known}
        df = df.groupby(df['name']).aggregate(
            aggregation_functions).reset_index()
        aggregation_functions = {'name': substringSieve}
        df = df.groupby(df['entity']).aggregate(
            aggregation_functions).reset_index().explode('name')

        df_result = df
        # df_result = concat([df_result, address_finder(
        #     df=df[df['entity'] == 'ADDRESS'],
        #     lookup=self.lookup)], ignore_index=True)
        return df_result
        df_result = concat([df_result, address_finder(
            df=df[df['entity'] == 'ADDRESS'],
            lookup=self.lookup)], ignore_index=True)
        """
        response = df.to_dict(orient='records')

        return response  # df_result


class LanguageDetector:
    def __init__(self, pretrained_lang_model, default_language='en'):
        pretrained_lang_model = pretrained_lang_model if pretrained_lang_model.startswith(
            '/') else f'{os.path.dirname(__file__)}/../../{pretrained_lang_model}'
        self.model = fasttext.load_model(pretrained_lang_model)
        self.default_language = default_language
        self.langid = LID()

    def detect_lang(self, text):
        # returns top 2 matching languages
        predictions = self.model.predict(text, k=2)
        try:
            language = predictions[0][0].replace('__label__', '')
            return self.langid.predict(text) if language == 'zh' else language
        except Exception:
            log.error(f'Error detecting language for {text}')
            return self.default_language


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def get_lazy_value(self, key):
        _, arg = self._raw_dict.__getitem__(key)
        return arg

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def json_prompt_base():

    result = {}
    for file in glob(f"{os.path.dirname(__file__)}/gpt_ner_json_prompts/*_prompt.json"):
        with open(file, "r", encoding="utf-8") as read_file:
            key = file.split('/')[-1].split('_prompt')[0]
            result[key] = json.load(read_file)
    return result


def adjust_path(params: DictConfig) -> None:
    for key in params.keys():
        if 'path' in key and any(word in key for word in ['model', 'datasets']):
            if not params[key].startswith('/') and params[key] not in ['bert-base-multilingual-cased']:
                params[key] = f'{os.path.dirname(__file__)}/../../{params[key]}'

    return params


@cache
def instantiate_cached(params):
    params = adjust_path(params)
    return instantiate(params)


@cache
def weaviate_client(url, key):
    import weaviate
    client = weaviate.Client(
        url=url,
        additional_headers={
            "X-OpenAI-Api-Key": key
        }
    )
    return client


@cache
def load_spacy(key):
    return spacy.load(name=key[0], disable=key[1])


@cache
def read_dataset(key):
    path = key[0]
    remove_non_parquet = key[1]
    path = path if path.startswith(
        '/') else f'{os.path.dirname(__file__)}/../../{path}'
    files = glob(rf'{path}/*')
    for file in files:
        extension = file.split('.')[-1]
        if extension != 'parquet':
            log.warning(f'File {file} is not a parquet file')
            if remove_non_parquet:
                os.remove(file)
                log.warning(f'Removed {file}')
            else:
                log.warning(f'Raising exception for {file}')
                raise TypeError(
                    f'File {file} is not a parquet file')
    return path


def read_df_from_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
        df = json_normalize(data)
        return df


@cache
def read_dataframe(
    input_path: str,
    input_header_names: str = None,
    max_records_to_process: int = -1,
    header: int = 0,
):
    df_all = DataFrame()
    df = DataFrame()
    input_path = input_path if input_path.startswith(
        '/') else f'{os.path.dirname(__file__)}/../../{input_path}'
    files = glob(rf'{input_path}')

    for file in files:
        extension = file.split('.')[-1]
        if extension in ['json']:
            df = read_df_from_json(file)
        elif extension in ['parquet']:
            df = read_parquet(file)
        elif extension in ['csv']:
            df = read_csv(file, header=header, names=input_header_names, warn_bad_lines=True,
                          error_bad_lines=False)
        elif extension in ['tsv']:
            df = read_csv(file, header=header, names=input_header_names, warn_bad_lines=True,
                          error_bad_lines=False, sep='\t')
        else:
            log.error(f'File extension {extension} not supported')
            raise TypeError("File extension not supported")

            # df = read_fwf(file, header=None, names=[params.text_column]) if extension in ['source'] else df
        if len(df) > 0 and max_records_to_process > 0:
            max_records_to_process = max_records_to_process
            df = df[:max_records_to_process] if max_records_to_process else df
        df_all = concat([df_all, df], ignore_index=True)
    df_all.path = input_path
    return df_all


@cache
def documents2dataframe(input_path: str, n_cores: int = None):
    input_path = input_path if input_path.startswith(
        '/') else f'{os.path.dirname(__file__)}/../../{input_path}'
    files = glob(rf'{input_path}')
    df_result = DataFrame()
    n_cores = n_cores or cpu_count()
    log.info(f'Extracting content {len(files)} files Using {n_cores} cores')
    with Pool(n_cores) as p:
        df_result = concat(list(
            tqdm(p.map(extract_content, files), total=len(files))), ignore_index=True)
    log.info(f"{df_result.shape[0]} text extracted from {len(files)} files")
    return df_result


def extract_content(file):

    df_result = DataFrame()
    extension = file.split('.')[-1]
    row = {}
    if extension == 'pdf':
        try:
            # creating a pdf file object
            pdfFileObj = open(file, 'rb')
            viewer = SimplePDFViewer(pdfFileObj)

            # printing number of pages in pdf file
            # log.info(viewer.metadata)
            row = viewer.metadata
            file_name = Path(file).stem
            row['article_number'] = file_name.split(
                '_')[0] if len(file_name.split('_')) > 1 else None
            title = file_name.split('_')[1] if len(
                file_name.split('_')) > 1 else file_name
            row['file_title'] = title.replace(' - Law360', '')
            # creating a page object
            for canvas in viewer:
                if canvas is not None:
                    page_strings = canvas.strings
                    row['image_count'] = len(
                        canvas.images) if canvas.images is not None else row.get('image_count', 0)
                    row['inline_image_count'] = len(
                        canvas.inline_images) if canvas.inline_images is not None else row.get('inline_image_count', 0)
                    row['form_count'] = len(
                        canvas.forms) if canvas.forms is not None else row.get('form_count', 0)
                    if row.get('text') is None:
                        if len(page_strings) > 2:
                            row['article_date'] = page_strings[0]
                            row['title'] = page_strings[1]
                            text = "".join(page_strings[2:])
                            url = text.split(' ')[0]
                            row['url'] = url
                            row['text'] = text.replace(url, '')
                            # link = text.substring(text.indexOf('http'), text.indexOf('http') + 100)

                # closing the pdf file object
            pdfFileObj.close()
        except TypeError as e:
            log.info(f"******* Error processing {file} {e}")
    elif extension == 'txt':
        try:
            file_name = Path(file).stem
            title = file_name.split('_')[1] if len(
                file_name.split('_')) > 1 else file_name
            row['file_title'] = title
            f = open(file, "r")
            content = f.read()
            # split order of content is important
            # order of content is title, text, published_date
            content = content.split('--')
            row['text'] = '--'.join(content[1:]).strip()
            content = content[0].split('\n\nBy')
            row['title'] = content[0].strip()
            row['author'] = content[1].split('(')[0].strip()
            row['published_date'] = content[1].split(
                '(')[1].split(')')[0].strip()
            f.close()
        except Exception as e:
            pass
    else:
        log.info(f"Processing extension {extension} not supported")

    df_result = concat([df_result, DataFrame(
        row, index=[0])], ignore_index=True)
    return df_result


@cache
def get_folder(path):
    path = path if path.startswith(
        '/') else f'{os.path.dirname(__file__)}/../../{path}'
    return path


def FieldSchemaConverter(field):
    value = field.pop('dtype', None)
    field.dtype = DataType[value] if value is not None else None
    return FieldSchema(**field)


@cache
def get_collection(params):
    if params["recreate"]:
        utility.drop_collection(params['name'])
    try:
        fields = [
            FieldSchemaConverter(x) for x in params.fields
        ]
        schema = CollectionSchema(
            fields=fields, description=params['description'])
        collection = Collection(name=params['name'], schema=schema)
        index_params = dict(params['index_params'])
        index_params['params'] = dict(index_params['params'])
        collection.create_index(
            field_name=params['field_name'], index_params=index_params)
        collection.load(replica_number=1)
        return collection
    except Exception as e:
        log.warning(e)
        collection = Collection(params['name'])  # Get an existing collection.
        collection.load(replica_number=1)
        return collection


class Cache:

    try:

        current_path = os.path.dirname(__file__)
        cfg = OmegaConf.load(f'{current_path}/config.yaml')

        try:
            import openai
            openai.api_type = cfg.services.openai.api_type
            openai.api_base = cfg.services.openai.api_base
            openai.api_version = cfg.services.openai.api_version
            openai.api_key = cfg.services.openai.access_key
        except (ImportError, ConfigAttributeError):
            log.warning("OpenAI not installed")
            openai = None
        try:
            from pymilvus import MilvusException, connections
            connections.connect(
                "default", host=cfg.services.milvus.host, port=cfg.services.milvus.port)
        except (ImportError, ConfigAttributeError):
            log.warning("Milvus not installed")
        except MilvusException as e:
            log.error(f"Milvus connection error {e}")

        try:
            import weaviate
            weaviate_client = weaviate_client(
                url=cfg.services.weaviate.url,
                # cfg.services.weaviate.openai_key,
                key=os.environ['openai_api_key']
            ) if 'openai_api_key' in os.environ else None
        except (ImportError, ConfigAttributeError):
            log.warning("Weaviate not installed")
            weaviate_client = None

        with open(f"{current_path}/hydra/job_logging/custom.yaml", 'r') as stream:
            logging_config = yaml.load(stream, Loader=yaml.FullLoader)

        setence_segementation_limit = cfg.setence_segementation_limit.inference if 'setence_segementation_limit' in cfg else 1000
        minimum_char_length_between_infered_blocks = cfg.signature_construction.minimum_char_length_between_infered_blocks if 'signature_construction' in cfg else 0
        regex_ner = RegexNER(
            conf=cfg.regex_ner) if 'regex_ner' in cfg else None
        services = cfg.services if 'services' in cfg else None
        search_result_fields = cfg.search_result_fields if 'search_result_fields' in cfg else None

        nergptconfig = cfg.nergptconfig if 'nergptconfig' in cfg else None
        max_tokens = cfg.max_tokens if 'max_tokens' in cfg else None

        language = cfg.language
        test_datasets = dict(
            cfg.test_datasets) if 'test_datasets' in cfg else {}
        constants = cfg.constants if 'constants' in cfg else None
        language_map = dict(cfg.language_map) if 'language_map' in cfg else {}
        spacy_languages = dict(cfg.spacy.languages).keys(
        ) if 'spacy' in cfg and cfg.spacy and cfg.spacy.languages else []
        valid_languages = list(
            cfg.valid_languages) if 'valid_languages' in cfg else []
        try:
            json_prompts_folder = get_folder(
                cfg.services.openai.json_prompts_folder)
        except Exception:
            log.info("json_prompts_folder not found")
            json_prompts_folder = None
        prompts = cfg.prompts if 'prompts' in cfg else None
        configs = cfg.configs if 'configs' in cfg else None
        commands = cfg.commands if 'commands' in cfg else None
        json_templaes = cfg.json_templaes if 'json_templaes' in cfg else None
        curators = cfg.curators if 'curators' in cfg else None
        json_prompt_base = json_prompt_base()

        with open(f"{current_path}/../conf/{cfg.regex_ner.name_finder_path}", 'r', encoding="utf-8") as f:
            # json object: countries, states, and cities
            countries = json.loads(f.read())

        if 'spacy' in cfg and cfg.spacy and cfg.spacy.languages:
            params = {}
            for key, value in cfg.spacy.languages.items():
                params[key] = (load_spacy, (value, cfg.spacy.disable))

            spacy_pipelines = LazyDict(params.items())

        if 'pipelines' in cfg and cfg.pipelines:
            params = {}
            for key, value in cfg.pipelines.items():
                params[key] = (instantiate_cached, value)

            pipelines = LazyDict(params.items())

        if 'models' in cfg and cfg.models:
            params = {}
            for key, value in cfg.models.items():
                params[key] = (instantiate_cached, value)

            models = LazyDict(params.items())
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        if 'tokenizers' in cfg and cfg.tokenizers:
            params = {}
            for key, value in cfg.tokenizers.items():
                params[key] = (instantiate_cached, value)

            tokenizers = LazyDict(params.items())

        if 'data_modules' in cfg and cfg.data_modules:
            params = {}
            for key, value in cfg.data_modules.items():
                params[key] = (instantiate_cached, value)

            data_modules = LazyDict(params.items())

        if 'dataframes' in cfg and cfg.dataframes:
            params = {}
            for key, value in cfg.dataframes.items():
                params[key] = (read_dataframe, value)

            dataframes = LazyDict(params.items())

        if 'datasets' in cfg and cfg.datasets:
            params = {}
            for key, value in cfg.datasets.items():
                # remove non parquet files
                params[key] = (read_dataset, (value, True))

            datasets = LazyDict(params.items())

        if 'documents' in cfg and cfg.documents:
            params = {}
            for key, value in cfg.documents.items():
                params[key] = (documents2dataframe, value)

            documents = LazyDict(params.items())

        if 'folders' in cfg and cfg.folders:
            params = {}
            for key, value in cfg.folders.items():
                params[key] = (get_folder, value)

            folders = LazyDict(params.items())

        if 'collections' in cfg and cfg.collections:
            params = {}
            for key, value in cfg.collections.items():
                params[key] = (get_collection, value)

            collections = LazyDict(params.items())
            collection_names = list(cfg.collections.keys())

    except Exception as e:
        log.info(e)
        log.info(f"Cache initialization failed {e}")
        raise e


# load all class variables
Cache()


def find_specific_service(table_name, text):
    df = Cache.dataframes[table_name]
    services = df[df['service'].str.contains('{' + f"{text}" + '}')]['service']
    return [service for service in services.values]


def stripper(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = stripper(v)
        if not v in (u'', None, {}, nan):
            new_data[k] = v
    return new_data


def get_service(endpoint):
    response = requests.get(endpoint)
    try:
        return DataFrame.from_dict(response.json())
    except:
        parts = endpoint.split('/')
        return DataFrame.from_dict({parts[-2]: parts[-1], 'detail': response.json()} if len(parts) > 2 else {'detail': response.json()})


def post_to_neo4j(post_value):
    url = Cache.services.neo4j.url
    message = {"statements": [{"statement": post_value}]}
    response = requests.post(url, json=message)
    return response
