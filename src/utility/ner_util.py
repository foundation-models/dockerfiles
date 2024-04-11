import json
import logging
import os
import re

from pandas import DataFrame, concat

log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def known(series):
    if len(series) == 1:
        return series.iloc[0]
    for x in series:
        if x != 'UNKNOWN':
            return x


def substringSieve(series):
    string_list = list(series)
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out


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

    def clean_web(self, w):
        """ Clean the given web_address removing useless pre-defined strings.

        Args:
            w (list): an extracted instance of web address: [start, end, text]

        Returns:
            list: cleaned web address instance.
        """
        for k, v in self.clean_web_dict.items():
            # TODO: need to revise the start or end index too, not only the text.
            x = w[2] if w[2] != '.' else w[2][:-1]
            x = x.replace(k, v)
            w[2] = x
        return w

    def get_entities(self, text, df=DataFrame()):
        """ Extract the named entities in the given text using regexes.

        Args:
            text (str): text to extract named entities from.
            df (DataFrame, optional): possible previous populated named entities in the text. Defaults to DataFrame().

        Returns:
            dictionary: regex-based extracted named entities.
        """

        emails = [[e.start(), e.end(), e.group(0)] for e in self.email_regex.finditer(text)]
        webs = [[w.start(), w.end(), w.group(0)] for w in self.web_regex.finditer(text)]
        linkedin = [[l.start(), l.end(), l.group(0)] for l in self.linkedin_regex.finditer(text)]
        
        # filter extracted linkedins.        
        linkedin = [l for l in linkedin if 'linkedin' in l[2] and l[2] != 'linkedin.com']
        linkedin = [l for l in linkedin if not any( [c in l[2] for c in ['@', 'malito']])]
        linkedin = [l[:-1] if l[-1] == '>' else l for l in linkedin]
        
        # filter the web addresses.
        webs = [w for w in webs if ((w not in linkedin) and (w not in emails))]
        webs = [w for w in webs if not any([c in w[2] for c in ['@', 'malito']])]

        df = concat([df, DataFrame(
            [{'start': item[0], 'end': item[1], 'entity': 'EMAIL', 'name': item[2], 'score': 1.0} for item in emails])], ignore_index=True)
        df = concat([df, DataFrame(
            [{'start': item[0], 'end': item[1], 'entity': 'LINKEDIN', 'name': item[2], 'score': 1.0} for item in linkedin])], ignore_index=True)
        df = concat([df, DataFrame(
            [{'start': item[0], 'end': item[1], 'entity': 'URL', 'name': item[2], 'score': 1.0} for item in webs])], ignore_index=True)
        
        response = df.to_dict(orient='records')

        return response #df_result

    def get_entities_old(self, text, df=DataFrame()):
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

        return response #df_result

def concat_near_by(df, text):
    start = df['start'].iloc[0].astype(int)
    end = df['end'].iloc[0].astype(int)
    result = {"start": [], "end": [], "name": []}
    for i in range(1, len(df)):
        if df['start'].iloc[i] - df['end'].iloc[i-1] < 4:
            end = df['end'].iloc[i].astype(int)
        else:
            result["start"].append(start)
            result["end"].append(end)
            result["name"].append(text[start:end])
            start = df['start'].iloc[i].astype(int)
            end = df['end'].iloc[i].astype(int)
    result["start"].append(start)
    result["end"].append(end)
    result["name"].append(text[start:end])
    return DataFrame(result)


def find_ner(item):
    text, df = item
    result = DataFrame()
    df["name"] = df[["start", "end"]].apply(
        lambda row: text[row['start']:row['end']], axis=1)
    groups = df.groupby('entity')
    result = DataFrame()
    for entity_name, group in groups:
        df_result = concat_near_by(df=group, text=text)
        df_result["entity"] = entity_name
        result = concat([result, df_result])
    return result
