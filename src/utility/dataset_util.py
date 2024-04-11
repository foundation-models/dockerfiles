import logging
import os
from functools import partial

from multiprocessing import Pool, cpu_count
from typing import Generator

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from fastapi import HTTPException

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm
from conf.cached import Cache
from utility.constants import CommonColumn, SpacyTags, TokenClassificationColumn

log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def tokenize_function(examples, tokenizer, text_column):
    try:
        return tokenizer(examples[text_column])
    except TypeError as e:
        cleanedList = ['' if i is None else i for i in examples[text_column]]
        return tokenizer(cleanedList)
    except Exception as e:
        log.error(e)
        raise e


# Passing a text column to tokenizer and be sure the number of tokens doesnt go above the max length
def align_dataset_on_model_max_length(datasets, tokenizer, text_column="text", remove_columns=["text"]):
    tokenized_datasets = datasets.map(partial(tokenize_function, tokenizer=tokenizer,
                                      text_column=text_column), batched=True, num_proc=cpu_count, remove_columns=remove_columns)
    lm_datasets = tokenized_datasets.map(
        partial(group_texts, tokenizer=tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    return lm_datasets


def group_texts(examples, tokenizer):
    block_size = tokenizer.model_max_length
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def train_test_splitter(df, group_by_column_name, validation_test_ratio, random_state):
    if validation_test_ratio == 0:
        return df, DataFrame()
    if group_by_column_name is not None:
        splitter = GroupShuffleSplit(
            test_size=validation_test_ratio, n_splits=2, random_state=random_state)
        split = splitter.split(df, groups=df[group_by_column_name])
        first_indices, second_indices = next(split)
        train_df = df.iloc[first_indices]
        test_df = df.iloc[second_indices]
        return train_df, test_df
    else:
        return train_test_split(df, test_size=validation_test_ratio, random_state=random_state)


def dataset_to_dataframe(processed_dataset, category, reconstruction_method=None):
    data = processed_dataset.get(category)
    if reconstruction_method:
        with Pool(cpu_count()) as p:
            result = pd.concat(list(
                tqdm(p.map(reconstruction_method, data), total=len(data))), ignore_index=True)
    else:
        result = DataFrame(data)[['Text', 'Labels']].rename(
            columns={"Text": "tokens", "Labels": "ner_tags"})
    return result


def map_by_email_id(df, id_column, tokenizer, language_column, text_column, token_column=None,
                    label_column=None, remove_columns=None, include_carriage_return=False, ner_tags_list=None,
                    max_token_per_segment=250,
                    use_single_process=False,
                    ):
    # keep only the impotant columns
    if remove_columns is not None and set(remove_columns).issubset(set(df.columns)):
        df = df.drop(columns=remove_columns)
    # remove all the rows with nan values
    df = df.dropna()
    # group by email id to be sure we don't mix data fron one email to another
    groups = df.groupby(id_column)
    # depending on the language if training a multilingual model we need to seaprate data based on language

    n_cores = 1 if use_single_process else min(groups.ngroups, cpu_count())
    with Pool(n_cores) as p:
        df = pd.concat(list(tqdm(p.map(partial(map_by_language,
                                               tokenizer=tokenizer,
                                               language_column=language_column,
                                               text_column=text_column,
                                               group_by_column_name=id_column,
                                               label_column=label_column,
                                               token_column=token_column,
                                               include_carriage_return=include_carriage_return,
                                               ner_tags_list=ner_tags_list,
                                               max_token_per_segment=max_token_per_segment,
                                               ), groups))), ignore_index=True)
    return df


def map_by_language(item, tokenizer, language_column, text_column, group_by_column_name,
                    token_column=None, label_column=None, include_carriage_return=False, ner_tags_list=None,
                    max_token_per_segment=250,
                    ):
    group_id, df = item
    result = DataFrame()

    if language_column in df.columns:
        groups = df.groupby(language_column)
        for language, group in groups:
            sentnce_segmentation_pipeline = Cache.spacy_pipelines[language]
            df = retrieve_tokens_ner_tags(df=group,
                                          tokenizer=tokenizer,
                                          text_column=text_column,
                                          label_column=label_column,
                                          include_carriage_return=include_carriage_return,
                                          ner_tags_list=ner_tags_list,
                                          sentnce_segmentation_pipeline=sentnce_segmentation_pipeline,
                                          max_token_per_segment=max_token_per_segment,
                                          )
            df[language_column] = language
            result = pd.concat([result, df], ignore_index=True)
        result[group_by_column_name] = group_id
        result = result.dropna()
    else:
        result = retrieve_tokens_ner_tags(df=df,
                                          tokenizer=tokenizer,
                                          text_column=text_column,
                                          token_column=token_column,
                                          label_column=label_column,
                                          include_carriage_return=include_carriage_return,
                                          ner_tags_list=ner_tags_list,
                                          sentnce_segmentation_pipeline=sentnce_segmentation_pipeline,
                                          max_token_per_segment=max_token_per_segment,
                                          )
    result[group_by_column_name] = group_id
    return result


def limit_numer_of_tags(tokens, ner_tags, ner_tags_list):
    limited_tokens = []
    limited_ner_tags = []
    for token, ner_tag in zip(tokens, ner_tags):
        if ner_tag in ner_tags_list:
            limited_tokens.append(token)
            limited_ner_tags.append(ner_tag)
    return limited_tokens, limited_ner_tags


def concat_result(result, tokens, tags):
    row = DataFrame([(tokens, tags)],
                    columns=[TokenClassificationColumn.tokens.value,
                             TokenClassificationColumn.tags.value
                             ]) if tags else \
        DataFrame(
            {TokenClassificationColumn.tokens.value: [np.array(tokens)]})
    result = pd.concat([result, row], ignore_index=True)
    return result


def retrieve_tokens_ner_tags(df, tokenizer,
                             text_column=None,
                             token_column=None,
                             label_column=None,
                             include_carriage_return=False,
                             ner_tags_list=None,
                             sentnce_segmentation_pipeline=None,
                             max_token_per_segment=250,
                             ):
    # this df contains only one language and one email id

    result = DataFrame()

    if text_column is not None and text_column in df.columns:

        for limit in range(int(0.1*max_token_per_segment), int(0.7*max_token_per_segment), int(0.1*max_token_per_segment)):
            # this is for text segmentation
            segments = get_train_segments(
                text_blocks=df[text_column],
                label_blocks=df[label_column],
                limit=limit,
                pipeline=sentnce_segmentation_pipeline
            )
            for segment in segments:
                result = concat_result(
                    result, segment['tokens'], segment['labels'])
        # token_count = TokenClassificationColumn.token_count.value
        # df[token_count] = df[text_column].map(tokenizer.tokenize).map(len)

        # model_max_length = tokenizer.model_max_length
        # tokens = []
        # tags = []
        # n_tokens = 0
        # index = 0
        # while( index < len(df) ):
        #     item = df.iloc[index]
        #     n_tokens += item[token_count]
        #     if n_tokens < model_max_length:
        #         this_token = item[text_column].split()
        #         tokens += this_token + [ControlCharacters.new_line.value] if include_carriage_return else this_token
        #         tags += [item[label_column]] * (len(this_token) +
        #                                 (1 if include_carriage_return else 0)) if label_column else []
        #         index += 1
        #     else:
        #         result = concat_result(result, tokens, tags)
        #         if( index > 0 and (item[token_count] + df.iloc[index-1][token_count]) < model_max_length):
        #             index -= 1
        #         elif(item[token_count] > model_max_length):
        #             for block in wrap(item[text_column], int(model_max_length/2),
        #                     drop_whitespace=False,
        #                     break_on_hyphens=False
        #                 ):
        #                 tokens = block.split()
        #                 tags = [item[label_column]] * len(tokens)
        #                 result = concat_result(result, tokens, tags)
        #             index += 1
        #         else:
        #             pass
        #         n_tokens = 0
        #         tokens = []
        #         tags = []
        # result = concat_result(result, tokens, tags) if tokens else result
    elif token_column is not None and token_column in df.columns:
        # This is for NER
        df = df[df[token_column] != '[\\n]']
        tokens = [x for x in df[token_column].values]
        ner_tags = [x for x in df[label_column].values]
        if ner_tags_list:
            tokens, ner_tags = limit_numer_of_tags(
                tokens, ner_tags, ner_tags_list)
        result[TokenClassificationColumn.tokens.value] = [tokens]
        result[TokenClassificationColumn.tags.value] = [ner_tags]
    return result


def dataset_to_dataframe(processed_dataset, category, reconstruction_method=None):
    data = processed_dataset.get(category)
    if reconstruction_method:
        with Pool(cpu_count()) as p:
            result = pd.concat(list(
                tqdm(p.map(reconstruction_method, data), total=len(data))), ignore_index=True)
    else:
        result = DataFrame(data)[['Text', 'Labels']].rename(
            columns={"Text": "tokens", "Labels": "ner_tags"})
    return result


def split_train_validation_test(processed_dataset, reconstruction_method=None):
    df_train = dataset_to_dataframe(processed_dataset, 'train',
                                    reconstruction_method=reconstruction_method)
    df_validation = dataset_to_dataframe(processed_dataset, 'validation',
                                         reconstruction_method=reconstruction_method)
    df_test = dataset_to_dataframe(processed_dataset, 'test',
                                   reconstruction_method=reconstruction_method)

    datasets_train_test = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_validation),
        "test": Dataset.from_pandas(df_test)
    })
    log.info(f'Created dataset with {len(datasets_train_test)} rows.')
    return datasets_train_test


def train_test_validaton_split(df,
                               dataset_output_path_prefix=None,
                               group_by_column_name=None,
                               validation_test_ratio=0.2,
                               validation_ratio_from_validation_test=0.5,
                               random_state=42
                               ) -> None:

    train, validaton_test = train_test_splitter(df=df,
                                                group_by_column_name=group_by_column_name,
                                                validation_test_ratio=validation_test_ratio,
                                                random_state=random_state
                                                )
    validaton, test = train_test_splitter(df=validaton_test,
                                          group_by_column_name=group_by_column_name,
                                          validation_test_ratio=validation_ratio_from_validation_test,
                                          random_state=random_state
                                          )
    if dataset_output_path_prefix:
        parent_folder = os.path.dirname(dataset_output_path_prefix)
        os.makedirs(parent_folder, exist_ok=True)
        Dataset.from_pandas(train, split='train').to_parquet(
            f'{dataset_output_path_prefix}.train.parquet')
        Dataset.from_pandas(validaton, split='validation').to_parquet(
            f'{dataset_output_path_prefix}.validation.parquet')
        Dataset.from_pandas(test, split='test').to_parquet(
            f'{dataset_output_path_prefix}.test.parquet')

    if group_by_column_name:
        train_set = set(train[group_by_column_name].unique())
        validation_set = set(validaton[group_by_column_name].unique())
        test_set = set(test[group_by_column_name].unique())
        if train_set.intersection(validation_set) or train_set.intersection(test_set) or validation_set.intersection(test_set):
            raise ValueError(
                'There is overlap between train, validation and test sets')
    return train, validaton, test


def filter_by_language(dataset, language, slice_name):
    df = dataset[slice_name].to_pandas()
    df = df[df[CommonColumn.language.value] == language].reset_index()
    dataset[slice_name] = Dataset.from_pandas(df, split=slice_name)


def build_ner_mappings(labels):
    id2label = {}
    label2id = {}
    for index, label in enumerate(labels):
        id2label[index] = label
        label2id[label] = index
    return id2label, label2id


def segment_sentence(sentence, limit):
    return [sentence[i:i+limit] for i in range(0, len(sentence), limit)]


def get_train_sentences_segments(text_blocks, label_blocks, limit, pipeline):
    """ tokenize the given blocks in order to not exceed the given limit.

    Args:
        text_blocks ([string]): list of block of string (body or signature or thread).
        label_blocks ([string]): list of corresponding label for the given text blocks.
        limit (integer): the upper limit of segment size in word count.
        pipeline (spacy pipeline): model to run the tokenizer

    Returns:
        [array]: two 2D arrays of words and corresponding labels
    """
    # result segments
    segments = []

    # general sentences and corresponding labels; 2D array for each.
    sentences, labels = [], []

    # iterate through the blocks and corresponding labels
    for (t, l) in zip(text_blocks, label_blocks):
        # run the pipeline on the text block.
        doc = pipeline(t)

        # iterate through the sentences of text block
        for s in doc.sents:

            # generate the list of tokens and labels for the sentence.
            s_tokens = [token.text for token in s]
            s_labels = [l] * len(s_tokens)

            # if the sentence is long, divide it into 2 pieces.
            # TODO: may need to revise this in future for a complete solution.
            if len(s_tokens) > limit:
                s_segments = segment_sentence(s_tokens, limit)

                # add all the segment of long sentence into the final lists.
                for seg in s_segments:
                    sentences.append(seg.copy())
                    labels.append([l]*len(seg))

                continue

            # append the tokens and corresponding labels to the list.
            sentences.append(s_tokens)
            labels.append(s_labels)

    # return sentences and labels
    return sentences, labels


def get_train_segments(text_blocks, label_blocks, limit, pipeline, overlapped_limit=None):
    """ segments the given blocks in order to not exceed the given limit.

    Args:
        text_blocks ([string]): list of block of string (body or signature or thread).
        label_blocks ([string]): list of corresponding label for the given text blocks.
        limit (integer): the upper limit of segment size in word count.
        pipeline (spacy pipeline): model to run the tokenizer

    Returns:
        [dictionary]: each entry: (segment words, segment tags)
    """
    # result segments
    segments = []

    # general sentences and corresponding labels; 2D array for each.
    sentences, labels = get_train_sentences_segments(
        text_blocks, label_blocks, limit, pipeline)

    overlapped_limit = overlapped_limit or int(limit * 1.5)

    # Iterate through the sentences and labels and segment them
    s_count = len(sentences)
    segment_words, segment_labels, segment_lenght = [], [], 0

    for i in range(s_count):

        # if adding current sentence exceeds the limit length for the segment, save the current segment and create a new one.
        if len(segment_words) + len(sentences[i]) > limit:

            # add the current segment to the results if it is not the first sentence.
            if i > 0:
                segments.append({'tokens': segment_words.copy(),
                                'labels': segment_labels.copy()})

                # if there is any previous sentence + its sum with the current one < limit, add it to the new segment for overlap.
                if len(sentences[i-1]) + len(sentences[i]) > overlapped_limit:
                    segment_words = sentences[i]
                    segment_labels = labels[i]
                    continue

            # add the previous sentence to the new segment.
            segment_words = sentences[i-1] + sentences[i]
            segment_labels = labels[i-1] + labels[i]

            continue

        # add the current sentence to the segment.
        segment_words += sentences[i]
        segment_labels += labels[i]

    # add the last segment to the results
    segments.append({'tokens': segment_words.copy(),
                    'labels': segment_labels.copy()})

    # return the segments
    return segments


def get_inference_sentences(text, limit, pipeline):

    # tokeniezed sentences and raw texts.
    sentences, indexes = [], []

    # run the pipeline on the text.
    doc = pipeline(text)

    # iterate through the sentences of text block.
    for s in doc.sents:

        # generate the list of tokens and labels for the sentence.
        s_tokens = [token for token in s]

        # if the sentence is long, divide it into 2 pieces.
        if len(s_tokens) > limit:
            s_segments = segment_sentence(s_tokens, limit)

            # add all the segment of long sentence into the final lists.
            for seg in s_segments:
                sentences.append(seg.copy())
                start = seg[0].idx
                end = seg[-1].idx + len(seg[-1])
                indexes.append((start, end))

            continue

        # append the tokens and corresponding labels to the list.
        sentences.append(s_tokens)
        start = s_tokens[0].idx
        end = s_tokens[-1].idx + len(s_tokens[-1])
        indexes.append((start, end))

    return sentences, indexes


def get_inference_segments(text, limit, pipeline):
    """ generate the segment of given text which the length of less than the limit

    Args:
        text (string): the given text block to be segmented
        limit (integer): the max length of each generated segment
        pipeline (nlp_pipeline): nlp pipeline to tokenize the text

    Returns:
        dictionary: list of segment, each segment = {tokens, start, end, text}
    """

    # get the senences and corresponding start_end indexes
    sentences, indexes = get_inference_sentences(text, limit, pipeline)

    # results.
    segments = []

    # current segment.
    segment, start, end = [], 0, 0

    # iterate through the sentence segments and their indexes.
    for sentence, index in zip(sentences, indexes):

        # if adding new sentence passes the limit, add previous segment to the results and build a new one.
        if len(segment) + len(sentence) > limit:
            segments.append(
                {'tokens': segment.copy(), 'start': start, 'end': end, 'text': text[start:end]})
            segment, start, end = sentence, index[0], index[1]
            continue

        # add the new sentence into the current segment.
        segment += sentence
        end = index[1]

    # add the last segment to the results.
    segments.append({'tokens': segment.copy(), 'start': start,
                    'end': end, 'text': text[start:end]})

    # return results.
    return segments


async def get_data_from_file(file_path: str) -> Generator:
    with open(file=file_path, mode="rb") as file_like:
        yield file_like.read()


def write_dataframe(output_file_name, df):
    extension = output_file_name.split(".")[-1]
    parent_dir = os.path.dirname(output_file_name)
    os.makedirs(parent_dir, exist_ok=True)
    if extension == 'csv':
        df.to_csv(output_file_name)
        media_type = 'text/csv'
    elif extension == 'parquet':
        df.to_parquet(output_file_name)
        media_type = 'application/octet-stream'
    else:
        raise HTTPException(
            status_code=404,
            detail="output_file_name must be csv or parquet")

    return output_file_name, media_type
