# This file contains the functions used for metrics calculation.
import pandas as pd
import numpy as np
from pandas import json_normalize, DataFrame
import datetime
import os
from conf.cached import Cache


def concat_entities(data_df: DataFrame, tag: str):
    """Concatenate the words of a given named entity type in the given DataFrame.

    Args:
        data_df (DataFrame): DataFrame containing the predictions.
        tag (str): The tag for a given named entity type.

    Returns:
        str: String of named entities for the given tag, separated by spaces.
    """
    # Get the rows with the given tag.
    df1 = data_df[data_df['entity'] == tag]

    if df1.empty or df1['name'].isnull().all():
        return ''

    entities_list = df1['name'].dropna().tolist()

    if len(entities_list) == 0:
        return ''
    elif len(entities_list) == 1:
        return entities_list[0].strip()
    else:
        return ' ; '.join(entities_list)


def post_process(response):
    """Process the response and generate the named entities.

    Args:
        response (dict): Result of the request to the API.

    Returns:
        list: List of dictionaries of {tag, entities}.
    """
    # Initialize results
    results = []

    # Populate the response into a DataFrame.
    preds_df = json_normalize(response)

    # Check if 'entity' column exists in preds_df
    if 'entity' not in preds_df.columns:
        return results

    # Iterate for each predicted tag in the results.
    for tag in set(preds_df['entity']):

        if tag == "UNKNOWN":
            continue

        # Concatenate the entities
        ents = concat_entities(preds_df, tag)

        # Add to the results.
        results.append({'tag': tag, 'entities': ents})

    return results


def calculate_weighted_average(performances: DataFrame, tagged_data: DataFrame):
    """ calculate the weighted average at entity level.

    Args:
        df_performance (DataFrame): performance metrics for all the named entities.
        df_tagged_data (DataFrame): DataFrame of the tagged data.

    Returns:
        DataFrame: Weighted average of the performance metrics.
    """

    tags_present = [tag for tag in Cache.ner_metrics.standard_tags_names if tag in tagged_data.columns]

    tags_weights = tagged_data[tags_present].count() / tagged_data[tags_present].count().sum()


    tags_weights = tags_weights.rename_axis('Entity').reset_index(name='WEIGHT')

    performances = pd.merge(performances, tags_weights, on='Entity')

    performances_average = performances[['PRECISION', 'RECALL', 'F1-MEASURE']].multiply(performances['WEIGHT'], axis="index").sum()
    
    non_zero_rows = performances[performances.notnull().all(axis=1)].shape[0]   

    return performances_average.to_frame().T.round(2), non_zero_rows



def format_ner_leaderboard_results(performances: DataFrame):
    """Format the leaderboard results for the NER task.

    Args:
        performances (DataFrame): performance metrics for all the named entities.
            this DataFrame should have the following columns:
            Entity, TP, FP, FN PRECISION, RECALL, F1-MEASURE

    Returns:        
        DataFrame: performance metrics for the named entities in ner leaderboard columns.
    """
    
    leader_board_rows = ['ADDRESS_CITY', 'ADDRESS_COUNTRY', 'ADDRESS_POSTAL_CODE', 'ADDRESS_STREET', 'EMAIL',
                         'ORGANIZATION', 'URL', 'PHONE_MOBILE', 'PHONE_OFFICE','JOB_TITLE', 'NAME_GIVEN', 'NAME_SURNAME']

    performance_filtered = performances[performances['Entity'].isin(leader_board_rows)]

    leaderboard_order = {entity: i for i, entity in enumerate(leader_board_rows)}
    performance_filtered = performance_filtered.sort_values('Entity', key=lambda s: s.map(leaderboard_order), ignore_index=True)

    performance_filtered[['PRECISION', 'RECALL', 'F1-MEASURE']] = performance_filtered[['PRECISION', 'RECALL', 'F1-MEASURE']].round(2)

    performance_filtered['LEADER_BOARD_VALUE'] = performance_filtered[['PRECISION', 'RECALL', 'F1-MEASURE']].apply(lambda x: f"{x[0]}/{x[1]}/{x[2]}", axis=1)

    return performance_filtered[['Entity', 'LEADER_BOARD_VALUE']]




def full_leaderboard_generation(performances: DataFrame, model_name: str, data_set_name: str, 
                                model_version: str, leaderboard_columns: list):

    leader_board = format_ner_leaderboard_results(performances)

    row_dict = {}

    for _, row in leader_board.iterrows():
        row_dict[str(row["Entity"])] = str(row["LEADER_BOARD_VALUE"]).replace("nan", "0.0")

    item_dictionary = {k: [v] for k, v in row_dict.items()}
    leader_board_row = pd.DataFrame(item_dictionary)

    leader_board_row["MODEL"] = f"{model_name}"
    leader_board_row["TEST_DATASET"] = f"{data_set_name}"
    leader_board_row["MODEL_VERSION"] = f"{model_version}"
    leader_board_row["LAST_UPDATE"] = datetime.datetime.now().strftime("%d/%m/%Y")

    leader_board_row = leader_board_row[leaderboard_columns]

    return leader_board_row


def generate_leaderboard(performances: DataFrame, model_name: str, data_set_name: str,
                         model_version: str, leaderboard_columns: list):

    performance_filtered = performances[performances['Entity'].isin(leaderboard_columns)]

    leaderboard_order = {entity: i for i, entity in enumerate(leaderboard_columns)}
    performance_filtered = performance_filtered.sort_values('Entity', key=lambda s: s.map(leaderboard_order), ignore_index=True)

    performance_filtered[['PRECISION', 'RECALL', 'F1-MEASURE']] = performance_filtered[['PRECISION', 'RECALL', 'F1-MEASURE']].round(2)

    row_dict = {}
    for _, row in performance_filtered.iterrows():
        row_dict[str(row["Entity"])] = f"{row['PRECISION']}/{row['RECALL']}/{row['F1-MEASURE']}".replace("nan", "0.0")

    item_dictionary = {k: [v] for k, v in row_dict.items()}
    leader_board_row = pd.DataFrame(item_dictionary)

    leader_board_row["MODEL"] = model_name
    leader_board_row["TEST_DATASET"] = data_set_name
    leader_board_row["MODEL_VERSION"] = model_version
    leader_board_row["LAST_UPDATE"] = datetime.datetime.now().strftime("%d/%m/%Y")

    for column in leaderboard_columns:
        if column not in leader_board_row.columns:
            leader_board_row[column] = '0.0/0.0/0.0'

    leader_board_row = leader_board_row[leaderboard_columns]

    return leader_board_row


def generate_weighted_leaderboard(performances: DataFrame, tagged_data: pd.DataFrame, model_name: str, data_set_name: str,
                                    model_version: str, weighted_columns_list: list):
    
    # select only the tags defined in the standard tags list
    tags_present = [tag for tag in Cache.ner_metrics.standard_tags_names if tag in tagged_data.columns]

    # calculate the relative weights for each tag
    tags_weights = tagged_data[tags_present].count() / tagged_data[tags_present].count().sum()

    # rename the index to Entity and the column to WEIGHT
    tags_weights = tags_weights.rename_axis('Entity').reset_index(name='WEIGHT')

    # merge the weights with the performances
    performances = pd.merge(performances, tags_weights, on='Entity')

    # calculate the weighted average of the performances
    performances_average = performances[['PRECISION', 'RECALL', 'F1-MEASURE']].multiply(performances['WEIGHT'], axis="index").sum()
    w_average_df = performances_average.to_frame().T.round(2)
    
    # calculate the number of non zero rows
    non_zero_rows = performances[performances.notnull().all(axis=1)].shape[0]

    # add the columns to the DataFrame
    w_average_df["MODEL"] = model_name
    w_average_df["ENTITIES_NUMBER"] = f"{non_zero_rows}/{len(Cache.ner_metrics.standard_tags_names)}"
    w_average_df["TEST_DATASET"] = data_set_name
    w_average_df["MODEL_VERSION"] = model_version
    w_average_df["LAST_UPDATE"] = datetime.datetime.now().strftime("%d/%m/%Y")

    return w_average_df[weighted_columns_list]



def get_date_as_str():
    """ get the date as string

    Args:
        date (datetime): date to be converted to string.

    Returns:
        str: date as string.

    """
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time to be used in the file name
    str_date = now.strftime("%Y_%m_%d_%H_%M")

    return str_date


# get the values from the json objetc avoiding keys not in the object
def extract_dict_keys(data_dict, key1, key2):
    if key1 not in data_dict or not data_dict[key1]:
        return ""
    return [d.get(key2, "") for d in data_dict[key1]]

# get the phone numbers from the json objetc avoiding keys not in the object
def extract_phones(data_dict, key1, key2_number, key3_tag, phone_type):
    if key1 not in data_dict or not data_dict[key1]:
        return ""
    return [d[key2_number] for d in data_dict[key1] if d.get(key3_tag) == phone_type and key2_number in d]


def parse_cmc_response(fields: dict):

    if fields is None:
        return []

    entities_response = [
        {"entity": "NAME_GIVEN", "name": extract_dict_keys(fields,"names","given")},
        {"entity": "NAME_SURNAME", "name": extract_dict_keys(fields,"names","surname")},
        {"entity": "NAME_PREFIX", "name": extract_dict_keys(fields,"names","prefix")},
        {"entity": "NAME_SUFFIX", "name": extract_dict_keys(fields,"names","suffix")},
        {"entity": "NAME_MIDDLE", "name": extract_dict_keys(fields,"names","middle")},
        {"entity": "JOB_TITLE", "name": extract_dict_keys(fields,"jobTitles","title")},
        {"entity": "ORGANIZATION", "name": extract_dict_keys(fields,"organizations","name")},
        {"entity": "EMAIL", "name": extract_dict_keys(fields,"emails","address")},
        {"entity": "PHONE_MOBILE", "name": extract_phones(fields,"phoneNumbers","number", "phoneType", "MOBILE")},
        {"entity": "PHONE_FAX", "name": extract_phones(fields,"phoneNumbers","number", "phoneType", "FAX")},
        {"entity": "PHONE_OFFICE", "name": extract_phones(fields,"phoneNumbers","number", "phoneType", "OFFICE")},
        {"entity": "ADDRESS_CITY", "name": extract_dict_keys(fields,"cities","name")},
        {"entity": "ADDRESS_POSTAL_CODE", "name": extract_dict_keys(fields,"postalCodes","code")},
        {"entity": "ADDRESS_STREET", "name": extract_dict_keys(fields,"addresses","line")},
        {"entity": "ADDRESS_COUNTRY", "name": extract_dict_keys(fields,"countries","name")},
        {"entity": "URL", "name": extract_dict_keys(fields,"webSites","url")}
    ]

    entites_final_response = []
    for entity in entities_response:
        names = entity["name"]
        if not names:
            entites_final_response.append({"entity": entity["entity"], "name": ""})
        elif isinstance(names, list):
            for name in names:
                entites_final_response.append({"entity": entity["entity"], "name": name})
        else:
            entites_final_response.append({"entity": entity["entity"], "name": names})

    return entites_final_response

def save_results(predictions_df, errors_df, performances_df,api_url_name, output_folder, language):
    """
    Save the predictions and errors to CSV files.

    Args:
        predictions_df (DataFrame): The predictions to save.
        errors_df (DataFrame): The errors to save.
        output_folder (str): The folder to save the files to.
    """
    # Save the predictions to a CSV file.
    predictions_file = os.path.join(output_folder, f"predictions_{api_url_name}_{language}_{get_date_as_str()}.csv")
    predictions_df.to_csv(predictions_file, index=False, encoding='utf-8')

    # Save the errors to a CSV file.
    errors_file = os.path.join(output_folder, f"errors_predictions_{api_url_name}_{language}_{get_date_as_str()}.csv")
    errors_df.to_csv(errors_file, index=False, encoding='utf-8')

    # Save the performances to a CSV file.
    performances_file = os.path.join(output_folder, f"performances_{api_url_name}_{language}_{get_date_as_str()}.csv")
    performances_df.to_csv(performances_file, index=False, encoding='utf-8')


def get_model_type(model_endpoint):
        
        """Get the model type from the model endpoint.
        
        Args:
            model_endpoint (str): The model endpoint.

        Returns:
            str: The model type.
        """
    
        # split the model endpoint by "/"
        model_endpoint_split = model_endpoint.split("/")
    
        # get the second element of the list
        model_type = model_endpoint_split[-2]
    
        # return the model type
        return model_type


def evaluate_results(tagged_data_df: pd.DataFrame, predictions_df: pd.DataFrame, model_name: str,  dataset_name: str, 
                     model_version: str, column_id_label: str , api_url: str):
    """
    Execute the evaluation of the predictions and generate the full_leader_board ang the weighted_leader_board dataframes.

    Args:
        tagged_data_df (DataFrame): The tagged data to evaluate.
        predictions_df (DataFrame): The predictions to evaluate.
        api_url (str): The URL of the API endpoint to call.

    Returns:
        DataFrame: The full_leader_board dataframe.
        DataFrame: The weighted_leader_board dataframe.
    """

    #filter the tagged_data_df only form email_id present in the predictions_df
    tagged_data_filtered = tagged_data_df[tagged_data_df[column_id_label].isin(predictions_df[column_id_label])]

    # Evaluate the predictions.

    performance_evaluation = evaluate_metrics(tagged_data_filtered, predictions_df, Cache.ner_metrics.standard_tags_names)

    #generate the leader boards dataframe
    full_leaderboard_df = generate_leaderboard(performance_evaluation, f"{model_name}-{get_model_type(api_url)}", dataset_name, model_version, Cache.ner_metrics.full_leaderboard_columns)

    # generate the weighted leader boards dataframe
    weighted_leaderboard_df = generate_weighted_leaderboard(performance_evaluation, tagged_data_filtered, f"{model_name}-{get_model_type(api_url)}", dataset_name, model_version, Cache.ner_metrics.leaderboard_weighted_averages_tags)

    return full_leaderboard_df, weighted_leaderboard_df, performance_evaluation


# JSON object to pass to the API for the request body.
body_json = {
    "emailMetadata": {
        "ccEmailAddresses": [], "bccEmailAddresses": [], "referenceMessageIds": [], "inReplyToMessageIds": [],
        "senderEmailAddress": { "address": "",  "displayName": ""},
        "fromEmailAddresses": [ {"address": "janelle.connell@fizzbin.com", "displayName": "Janelle Connell"  }],
        "replyToEmailAddresses": [ {"address": "janelle.connell@fizzbin.com", "displayName": "Janelle Connell"}],
        "toEmailAddresses": [ {"address": "bschelp@gwabbit.com", "displayName": "Brian Schelp"}],
        "subject": "GenerateEmls",
        "sentDate": 1538123341000,
        "messageId": "2012744708",
        "receivedByEmailAddress": "bschelp@gwabbit.com"
    },
    "emailBody": { "htmlBody": "",  "textBody": "" },
    "responseDataTypes": [ "EMAIL_METADATA", "SIGNATURE_BLOCK", "SIGNATURE_FIELDS"],
    "preferredBodyType": "TEXT_ONLY", "maxSignatures": 5
}

# JSON object to pass to the API for the request signature.
signature_json = {
    "emailMetadata": { "ccEmailAddresses": [], "bccEmailAddresses": [], "referenceMessageIds": [], "inReplyToMessageIds": [],
        "senderEmailAddress": { "address": "", "displayName": "" },
        "fromEmailAddresses": [ { "address": "janelle.connell@fizzbin.com", "displayName": "Janelle Connell"}],
        "replyToEmailAddresses": [{"address": "janelle.connell@fizzbin.com", "displayName": "Janelle Connell"}],
        "toEmailAddresses": [ {"address": "bschelp@gwabbit.com", "displayName": "Brian Schelp" }],
        "subject": "GenerateEmls",
        "sentDate": 1538123341000,
        "messageId": "2012744708",
        "receivedByEmailAddress": "bschelp@gwabbit.com"
    }
}


def get_json_request_signature(text):
    json_object = signature_json.copy()
    json_object["signatureBlock"] = text
    return (json_object)


def get_json_request_body(text):
    json_object = body_json.copy()
    json_object["emailBody"]["textBody"] = text
    return (json_object)


def create_html_page(general_results : pd.DataFrame, full_leaderboard_df : pd.DataFrame, 
                     full_weigthed_leaderboard_df : pd.DataFrame, full_detail_performance : pd.DataFrame, 
                     output_folder: str):

    """Create the html page with the results of the metrics calculation
        
        Args:
            general_results (list): list of dictionaries with the results of the metrics calculation
            full_leaderboard_df (pandas.DataFrame): dataframe with the full leaderboard results
            full_weigthed_leaderboard_df (pandas.DataFrame): dataframe with the weighted leaderboard results
            output_folder (str): path to the folder where the html page will be saved

        Returns:
            None
        """
    # create the html page
    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Endpoint NER metrics calculation</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
    </head>
    <body>
        <div class="container">
            <div class="row">
                <div class="col">
                    <h2>Rows proccesed</h2>
                </div>
            </div>
            <div class="row">
                <div class="col">
                    <h2>Report</h2>
                    <table class="table table-bordered table-striped">
                        <thead>
                            <tr>
                                <th>Endpoint</th>
                                <th>Number of predictions</th>
                                <th>Number of errors</th>
                            </tr>
                        </thead>
                        <tbody>
    """

    for result in general_results:
        html_page += f"""
                            <tr>
                                <td>{result["api_url"]}</td>
                                <td>{len(result["predictions"])}</td>
                                <td>{len(result["errors"])}</td>
                            </tr>
        """

    html_page += f"""
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="row">
                <div class="col-md-12">
                    <h2>Weighted leaderboard</h2>
                    {full_weigthed_leaderboard_df.to_html(classes="table table-bordered table-striped")}
                </div>
            </div>
            <div class="row">
                <div class="col-md-12">
                    <h2>Full leaderboard</h2>
                    {full_leaderboard_df.to_html(classes="table table-bordered table-striped")}
                </div>
            </div>
    """

    # add the full detail for all entities generated by evaluate method for all the endpoints as a table from the full_detail_performance dataframe
    html_page += f"""
            <div class="row">
                <div class="col-md-12">
                    <h2>Full detail</h2>
                    {full_detail_performance.to_html(classes="table table-bordered table-striped")}
                </div>
            </div>  
    """

    html_page += """
        </div>
    </body>
    </html>
    """

    # save the html page
    with open(os.path.join(output_folder, f"results_{get_date_as_str()}.html"), "w") as f:
        f.write(html_page)


def data_transformation(data_df):
    """Transform the data of the dataframe to be able to compare with the predictions
        
        Args:
            data_df (pandas.DataFrame): dataframe with the data to be transformed

        Returns:
            pandas.DataFrame: dataframe with the transformed data
        """
    disallowed_values = [None, "None",[], "[]", np.nan, "NaN", "NAN", "nan", "Null", "null"]

    def custom_replace(x):
        if x in disallowed_values:
            return ""
        else:
            return x
    
    try:
        data_df = data_df.applymap(custom_replace)

        for column in data_df.columns:
            if column not in Cache.ner_metrics.standard_tags_names:
                continue

            data_df[column] = data_df[column].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

            # Continue with the rest of the transformation
            data_df[column] = data_df[column].apply(lambda x: " ; ".join(x) if isinstance(x, list) else x)
            data_df[column] = data_df[column].apply(lambda x: " ; ".join(x[1:-1].split(",")) if x.startswith("[") and x.endswith("]") else x)
            data_df[column] = data_df[column].apply(lambda x: x.strip())
            data_df[column] = data_df[column].apply(lambda x: "" if x.isspace() else x)

    except Exception as e:
        print(f"Error performing the data transformation: {e}")
        return None    

    return data_df


def validate_data_format(data_df):
    """Validate the data format of the dataframe
        
        Args:
            data_df (pandas.DataFrame): dataframe with the data to be validated

        Returns:
            bool: True if the data format is valid, False otherwise
        """
    try:

        invalid_columns = set(data_df.columns) - set(Cache.ner_metrics.standard_tags_names + Cache.ner_metrics.standard_email_data)
        if invalid_columns:
            raise Exception(f"The dataframe contains columns that are not allowed: {invalid_columns}")

        # get only the list of tags columns that are present in the Cache.ner_metrics.standard_tags_names
        tags_columns = list(set(data_df.columns).intersection(set(Cache.ner_metrics.standard_tags_names)))

        disallowed_values = [None, "None",[], "[]", np.nan, "NaN", "NAN", "nan", "Null", "null"]

        # Validate that only allowed values are present in the cells
        if data_df[tags_columns].isin(disallowed_values).any().any():
            raise Exception("The dataframe contains values that are not allowed: Value: None, [], '[]', NaN, 'nan', Null, 'null'")

        # Validate that the values don't have the format [ any text ] for the columns defined in the STANDARD_COLUMNS_NAMES list
        if data_df[tags_columns].applymap(lambda x: isinstance(x, str)).all().all() and data_df[tags_columns].applymap(lambda x: x.strip().startswith("[") and x.strip().endswith("]")).any().any():
            raise Exception("The dataframe contains values that are not allowed: Value: array format [ any text ]")

        # Validate that not only one or more white spaces are present in the values for the columns defined in the STANDARD_COLUMNS_NAMES list
        if data_df[tags_columns].apply(lambda x: x.str.isspace()).any().any():
            raise Exception("The dataframe contains values that are not allowed: Value: only one or more white spaces")

        # Validate that multiple values are only separated by " ; " for the columns defined in the STANDARD_COLUMNS_NAMES list
        if data_df[tags_columns].apply(lambda x: x.str.contains(" , | : ")).any().any():
            raise Exception("The dataframe contains values that are not allowed: values not separated by ' ; '")

    except Exception as e:
        print(f"Error validating the data format: {e}")
        return False

    return True


def evaluate_metrics(true_df, predicted_df, tags):
    """Evaluate the metrics for the predicted tags

    Args:
        true_df (pandas.DataFrame): dataframe with the true tags
        predicted_df (pandas.DataFrame): dataframe with the predicted tags
        tags (list): list of tags to evaluate

    Returns:
        Pandas Dataframe: dataframe with the metrics for each tag
    """

    tag_counts = {tag: {'TP': 0, 'FP': 0, 'FN': 0} for tag in tags}

    # create a set of all the tags present in the true_df and the predicted_df only and also in the tags list
    tags = set(tags) & set(true_df.columns) & set(predicted_df.columns)

    for tag in tags:

        # if the tag is not present in the tryre_df dataframe, then skip it
        true_col = true_df[tag]
        pred_col = predicted_df[tag]

        for true, pred in zip(true_col, pred_col):
            true = '' if pd.isna(true) or true is None else true
            pred = '' if pd.isna(pred) or pred is None else pred

            true_set = set(map(str.strip, str(true).split(' ; '))) - {''}
            pred_set = set(map(str.strip, str(pred).split(' ; '))) - {''}

            if true_set or pred_set:
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                tag_counts[tag]['TP'] += tp
                tag_counts[tag]['FP'] += fp
                tag_counts[tag]['FN'] += fn

    metrics = {}
    
    for tag in tags:
        tp, fp, fn = tag_counts[tag]['TP'], tag_counts[tag]['FP'], tag_counts[tag]['FN']
        precision = 100* tp / (tp + fp) if tp + fp > 0 else 0
        recall = 100* tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        metrics[tag] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'PRECISION': precision,
            'RECALL': recall,
            'F1-MEASURE': f1_score
        }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Entity'})

    return metrics_df[['Entity','TP', 'FP', 'FN', 'PRECISION', 'RECALL', 'F1-MEASURE']]