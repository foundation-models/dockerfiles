import email
import logging
import os
import sys
import re
from pandas import DataFrame

import numpy as np
import pandas as pd
import torch
from conf.cached import Cache
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utility.constants import (Flag, LabelColumns, PipelineNames, PipelineTasks,
                               SignatureColumn, SignatureLabels)
from utility.dataset_util import get_inference_segments
from utility.ner_util import find_ner
from utility.named_finder import (NamedFinder, extract_locations)

from utility.schemas import (Address, City, Country, CountryDivision, Email,
                             JobTitle, Name, Organization, ParseStatusType,
                             PhoneNumber, PostalCode, SenderEmailDataEntry,
                             SenderEmailDataSourceTypes, SignatureBlock,
                             SignatureBlockResponse, SignatureFields,
                             SignatureResponse, WebSite)

log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def get_signature(text, language, create_response: bool = True):

    segmentation_pipeline = Cache.pipelines[f"{language}_{PipelineNames.segmentation_pipeline.value}"]
    spacy_pipeline = Cache.spacy_pipelines[language]
    df_result = splitwise_segment_signature(
        text=text,
        spacy_pipeline=spacy_pipeline,
        hf_pipeline=segmentation_pipeline,
        setence_segementation_limit=Cache.setence_segementation_limit,
        minimum_char_length_between_infered_blocks=Cache.minimum_char_length_between_infered_blocks
    )
    text_response = df_result[SignatureColumn.signature.value].values[0]
    beforesignature, finalchunk, aftersignature, minoffset, maxoffset = \
        process_signature(text, text_response)

    # temporary patching this issue, later we should fix this bug
    if text.count('\n') < 3:
        finalchunk = text_response

    torch.cuda.empty_cache()
    if create_response:
        return SignatureBlockResponse(
            body=beforesignature,
            signature=finalchunk,
            tail=aftersignature,
            signature_start_index=minoffset,
            signature_end_index=maxoffset
        )
    else:
        return beforesignature, finalchunk, aftersignature, minoffset, maxoffset

def get_named_entities_ml(text: str, language: str, create_response: bool = True):
    """ parse the signature using the NER pipeline using the signature text passed as input.

    Args:
        text (str): text to extract the named entities from.
        language (str): the language of given text.
        create_response (bool, optional): whether generate. Defaults to True.

    Returns:
        dictionary: dictionary of named entities
    """
    ner_pipeline = Cache.pipelines[f"{language.lower().strip()}_{PipelineNames.ner_pipeline.value}"]
    parser_pipeline = Cache.spacy_pipelines[language.lower().strip()]
    df_result = splitwise_segment_signature(text=text,
                                            mode=PipelineTasks.ner,
                                            spacy_pipeline=parser_pipeline,
                                            hf_pipeline=ner_pipeline,
                                            setence_segementation_limit=Cache.setence_segementation_limit,
                                            minimum_char_length_between_infered_blocks=Cache.minimum_char_length_between_infered_blocks
                                            )
    response = df_result.to_dict(orient='records')
    """
    if create_response:
        # response = json.loads(json.dumps(ner_parse_response(response)))
        response = parse_response(
            ner_entities=response, signature_block=text)
    """
    torch.cuda.empty_cache()
    return response

def get_named_entities(text: str, language: str, json_request_metadata: dict, create_response: bool = True):
    """ extract the named entities from the given string considering the language.

    Args:
        text (str): text to extract the named entities from.
        language (str): the language of given text.
        json_request_metadata (dict): meta data of request.
        create_response (bool, optional): whether generate. Defaults to True.

    Returns:
        dictionary: dictionary of named entities
    """
    
    # get the ML results.
    ml_results = get_named_entities_ml(text, language, create_response)
    
    # get the regex generated named entities.
    regex_results = Cache.regex_ner.get_entities(text)
    
    # get the generated named entiies by the Named Finder.
    nf_results = extract_locations(Cache.countries, text)
    
    # aggregate the results.
    results = aggregate_ner(ml_results, regex_results, nf_results, language)
    
    # post-process extracted named entities to filter useless ones. 
    results = post_process_ner(text, language, results)
    
    # if it is asked to generate the response in the particular format.
    if create_response:
        # response = json.loads(json.dumps(ner_parse_response(response)))
        results = populate_signature_response(
                entities=results, signature_block=text, json_request_metadata=json_request_metadata)
        
    return results

def aggregate_ner(ml_results, regex_results, nf_results, language):
    """ aggregate the results of machine learning, regex, and named_finder. 

    Args:
        ml_results (dictionary): extracted named entities by machine learning [start, end, entity, name].
        regex_results (dictionary): extracted named entities by the regex [start,end, entity, name].
        nf_results (dictionary): extracted named entities by named finder.

    Returns:
        dictionary: aggregated results.
    """
    aggregated_results = []
    
    # add the extracted values for the pre-defined ML named entities.
    aggregated_results = [item for item in ml_results if item['entity'] in Cache.cfg.regex_ner.aggregate.ml]

    # add all extracted named entities by regex.
    aggregated_results += [item for item in regex_results if item['entity'] in Cache.cfg.regex_ner.aggregate.regex]
        
    # add the extracted named entities by named finder to the results.
    tag_mapping = {
        "city": "ADDRESS_CITY", 
        "state": "ADDRESS_COUNTRY_DIVISION",
        "country":"ADDRESS_COUNTRY"
        }
    
    # get the list of first name, last name, and organization.
    names_orgs = [item['name'] for item in aggregated_results if item['entity'] in ["NAME_GIVEN", "NAME_MIDDLE", "NAME_SURNAME", "ORGANIZATION"]]
    
    # if the language is not English, use the results of ML for city, state, and country
    if language != 'en':
        aggregated_results += [item for item in ml_results if item['entity'] in tag_mapping.values()]
    
    # language is English, use the results of Named Finder for city, state, and country.
    else: 
        
        for tag in Cache.cfg.regex_ner.aggregate.nf:
        
            # lower the tag to be sync with the named_finder results.
            tag = tag.lower()
            
            # TODO: need to revise this. 
            if type(nf_results) is dict and tag in nf_results.keys():
                #recognized = [nf_results[tag][0]]
                recognized = [nf_results[tag]]
                        
            elif type(nf_results) is list:
                #recognized = [inst[tag][0] for inst in nf_results if tag in inst.keys()]
                recognized = [inst[tag] for inst in nf_results if tag in inst.keys() and len(inst[tag])>0]
            
            else:
                recognized = []
            
            # if there is no NF recognized.
            if len( recognized) == 0 or len(recognized[0]) == 0:
                continue
            
            # add the recognized ones to the results.
            aggregated_results += [{'start': item[-1][0], 
                                               'end': item[-1][1],
                                               'entity': tag_mapping[tag],
                                               'name': item[0],
                                               'score': 1.0} 
                                   for item in recognized 
                                   if (item[0] not in names_orgs) and 
                                   (item[0] not in Cache.cfg.regex_ner.aggregate.blacklist)]

    # return the aggregated results.
    return aggregated_results

def post_process_ner(text, language, entities):
    """ Post-process extracted named entities to filter useless one; e.g., those ones which are not a full token.
    Args:
        text (str): the text of signaure.
        language (str): the language of signature.
        entities (dictionary): extracted named entities [start,end, entity, name, score].

    Returns:
        dictionary: filtered results.
    """
    
    need_be_whole_token = ["NAME_GIVEN", "NAME_MIDDLE", "NAME_SURNAME", "NAME_PREFIX", "NAME_SUFFIX", 
                           "JOB_TITLE", "ORGANIZATION", "ADDRESS_STREET", "ADDRESS_POSTAL_CODE"]
    
    phone_tags = ["PHONE_MOBILE", "PHONE_OFFICE", "PHONE_HOME", "PHONE_FAX"]
    
    # cleaned entities.
    cleaned_entities = []

    # remove the end .,:;
    separators = (',', '.', ';', ':')
    for item in entities:
        if item['name'].endswith(separators):
            item['end'] -= 1
            item['name'] = item['name'][:-1]
        cleaned_entities.append(item)
    
    # No_New_Line: There is no Named Entity presented in 2 lines, so no \n in the middle of any named entity. 
    no_new_line = lambda x : len(x.strip(" "))<2 or x.strip(" ")[1:-1].find('\n') == -1
    results_0 = [item for item in cleaned_entities if no_new_line(item['name'])]
    
    # no_phone_or_full_phone: if the length of extracted phone/fax number is less than a threshold, remove it.
    no_phone_or_full_phone = lambda x: x['entity'] not in phone_tags or len(re.sub('\D', '', x['name'])) > 8
    results = [item for item in results_0 if no_phone_or_full_phone(item)]    

    # if the language is not English, do not apply remaining rules.
    if language != 'en':
        return results
    
    # complete_Tokens: the tokens of an entity are complete tokens.
    #complete_tokens = lambda x,text: set(x.strip().split(" ")).issubset(text)
    complete_tokens = lambda x,text: set(re.split(' |\n|:',x)).issubset(text)
    
    # split the text using space.
    tokens = re.split(' |\n|,|:|;', text)
        
    # select entities which all rules are applicable on.
    results = [item for item in results if complete_tokens(item['name'], tokens)]
    
    return results

def find_signature(item,
                   label_column,
                   minimum_char_length_between_infered_blocks=0,
                   starting_index=0,
                   signature_label=SignatureLabels.signature.value,
                   ):

    text, df = item
    result = DataFrame()
    try:
        df_sig = df[df[label_column] == signature_label]
        if len(df_sig) == 1:
            start = df_sig['start'].iloc[0].astype(int)
            end = df_sig['end'].iloc[0].astype(int)
            result[SignatureColumn.body.value] = [text[:start]]
            result[SignatureColumn.signature.value] = [text[start:end]]
            result[SignatureColumn.tag.value] = Flag.perfect_match.value
            result[SignatureColumn.start_signature.value] = [
                start + starting_index]
            result[SignatureColumn.end_signature.value] = [
                end + starting_index]
        elif len(df_sig) == 0:
            result[SignatureColumn.body.value] = [text]
            result[SignatureColumn.signature.value] = ['']
            result[SignatureColumn.tag.value] = Flag.no_match.value
        else:
            df_sig.loc[:, 'diff'] = (
                (df_sig['start'].shift(-1) - df_sig['end']).fillna(0)).astype(int)
            start = df_sig.iloc[0]['start'].astype(int)
            end = df_sig.iloc[-1]['end'].astype(int)
            result[SignatureColumn.body.value] = [text[:start]]
            result[SignatureColumn.signature.value] = [text[start:end]]
            result[SignatureColumn.tag.value] = Flag.imperfect_match.value

        return result
    except Exception as e:
        log.error(e)


def splitwise_segment_signature(spacy_pipeline, hf_pipeline, text,
                                mode=None,
                                setence_segementation_limit=100,
                                label_column=LabelColumns.single_processing.value,
                                signature_label=SignatureLabels.signature.value,
                                minimum_char_length_between_infered_blocks=0,
                                ):
    result = DataFrame()

    segments = get_inference_segments(
        text, setence_segementation_limit, spacy_pipeline)
    for segment in segments:
        content = segment["text"]
        if content and len(content) > 0:
            ner_results = hf_pipeline(content)
            df = DataFrame(ner_results)
            df_result = find_ner(item=(content, df)) if mode == PipelineTasks.ner \
                else find_signature(item=(content, df),
                                    label_column=label_column,
                                    starting_index=segment["start"],
                                    signature_label=signature_label,
                                    minimum_char_length_between_infered_blocks=minimum_char_length_between_infered_blocks
                                    )
            #df_result = Cache.regex_ner.get_entities(
                #df=df_result, text=content) if mode == PipelineTasks.ner else df_result
            result = pd.concat([result, df_result], ignore_index=True)

    return result


def splitwise_segment_signature_old(text,
                                    the_pipeline,
                                    minimum_char_length_between_infered_blocks=0,
                                    signature_label=SignatureLabels.signature.value,
                                    label_column=LabelColumns.single_processing.value
                                    ):
    result = DataFrame()
    i = 0
    while (i < len(text)):
        content = text[i:i+300]
        ner_results = the_pipeline(content)
        df = DataFrame(ner_results)
        df_result = find_signature(item=(content, df),
                                   label_column=label_column,
                                   starting_index=i,
                                   signature_label=signature_label,
                                   minimum_char_length_between_infered_blocks=minimum_char_length_between_infered_blocks
                                   )
        result = pd.concat([result, df_result], ignore_index=True)
        i += 150
    if len(result) > 1 and "end_signature" in result.columns:
        filter = result["end_signature"].notna()
        if filter.any():
            dfx = result[filter]
            dfx['diff'] = (dfx['start_signature'].shift(-1) -
                           dfx['end_signature']).fillna(dfx['end_signature']).astype(int)
            start = dfx.iloc[0]['start_signature'].astype(int)
            end = dfx.iloc[-1]['end_signature'].astype(int)
            for index in range(len(dfx)-1, 0, -1):
                if dfx.iloc[index-1]['diff'] < minimum_char_length_between_infered_blocks:
                    start = dfx.iloc[index-1]['start_signature'].astype(int)
                else:
                    break
            df_result[SignatureColumn.signature.value] = [text[start:end]]
            df_result[SignatureColumn.body.value] = [text[:start]]
    else:
        df_result = DataFrame()
        df_result[SignatureColumn.body.value] = [text]
        df_result[SignatureColumn.signature.value] = ['']
    return df_result


def remove_email_header(email_t: str):
    """ extract email body by removing its header.

    Args:
        email_t (str): the whole raw email.

    Returns:
        str: the body of email
    """
    email_eml = email.message_from_string(email_t)
    if email_eml.is_multipart():
        for part in email_eml.walk():
            payload = part.get_payload(decode=True)  # returns a bytes object
            strtext = payload.decode('utf-8', 'ignore')  # utf-8 is default
            return strtext
    else:
        payload = email_eml.get_payload(decode=True)
        strtext = payload.decode('utf-8', 'ignore')
        return strtext

def populate_entity(Entity_object: object, key_name: str, value_name: str, fields: dict, **kwargs):
    """ This method receives an object class reference and call the constructor dinamicallu using the value name object
    attribute passed as text and redd the calue from the fields dictionary using the key name passed as text and
    iterate to create a list of objects returning the list if any object is created or None if no object is created

    Args:
        Entity_object (object): Object class reference
        key_name (str): Key name to get the value from the dictionary
        value_name (str): Value name to create the object
        fields (dict): Dictionary to get the value
        **kwargs: Additional arguments to pass to the object constructor
    Returns:
        list: List of objects or None
    """

    object_list = [ Entity_object(**{value_name: str(item["name"]), **kwargs})
                   if item else None for item in fields.get(key_name, [""])]
    
    return object_list if any(object_list) else None

def populate_phones(entity_fields:dict):
    """populate various phone numbers from the extracted entities.

    Args:
        entity_fields (dict): dictionary of entities.

    Returns:
        list: list of populated phone numbers.
    """
    
    phone_mobile = populate_entity( PhoneNumber, "PHONE_MOBILE", "number", entity_fields, phoneType="MOBILE")
    phone_office = populate_entity( PhoneNumber, "PHONE_OFFICE", "number", entity_fields, phoneType="OFFICE")
    phone_home = populate_entity( PhoneNumber, "PHONE_HOME", "number", entity_fields, phoneType="HOME")
    fax = populate_entity( PhoneNumber, "PHONE_FAX", "number", entity_fields, phoneType="FAX")
    
    phones = phone_mobile if phone_mobile else []
    phones += phone_office if phone_office else []
    phones += phone_home if phone_home else []
    phones += fax if fax else []
    
    return phones if len(phones)>0 else None
    
def generate_response_metadata(json_request_metadata: dict, emails_data: list[Email]):
    """ Generates the response metadata from the request metadata.
    
    Args:
        json_request_metadata (dict): The request metadata.

    Returns:
        dict: The response metadata.
    """

    if not json_request_metadata:
        return []
    
    senderEmailAddress = json_request_metadata.get("senderEmailAddress", None)
    fromEmailAddresses = json_request_metadata.get("fromEmailAddresses", [])
    replyToEmailAddresses = json_request_metadata.get("replyToEmailAddresses", [])

    emailsData = []

    if senderEmailAddress:
        if "address"in senderEmailAddress:
            emailsData.append({"address": senderEmailAddress["address"], "displayName": senderEmailAddress["displayName"], "sourceTypes": SenderEmailDataSourceTypes.SENDER.value})

    if fromEmailAddresses:
        for email in fromEmailAddresses:
            if "address" in email:
                emailsData.append({"address": email["address"], "displayName": email["displayName"], "sourceTypes": SenderEmailDataSourceTypes.FROM.value})

    if replyToEmailAddresses:
        for email in replyToEmailAddresses:
            if "address" in email:
                emailsData.append({"address": email["address"], "displayName": email["displayName"], "sourceTypes": SenderEmailDataSourceTypes.REPLY_TO.value})

    if emails_data:
        if type(emails_data) is list:
            for email in emails_data:
                if email.address != "":
                    emailsData.append({"address": email.address, "displayName": "", "sourceTypes": SenderEmailDataSourceTypes.SIGNATURE_EMAIL_ADDRESS.value})

    senderEmail = {}

    for email in emailsData:
        if "address" in email and email["address"] != "":
            if email["address"] in senderEmail:
                senderEmail[email["address"]][0]["sourceTypes"].append(email["sourceTypes"])
                if senderEmail[email["address"]][1]["name"] == "":
                    senderEmail[email["address"]][1]["name"] = email["displayName"]
                                
            else:
                senderEmail[email["address"]] = []
                senderEmail[email["address"]].append({"sourceTypes" : [email["sourceTypes"]]})
                senderEmail[email["address"]].append({"name" : email["displayName"]})

    senderEmailData = []
    
    for address, values in senderEmail.items():
        if SenderEmailDataSourceTypes.SENDER.value in values[0]["sourceTypes"] or SenderEmailDataSourceTypes.FROM.value in values[0]["sourceTypes"]:
            senderEmailData.append(
                SenderEmailDataEntry(
                    address=address,
                    name=values[1]["name"],
                    confidenceScore=5,
                    sourceTypes=values[0]["sourceTypes"]
                )
            )

    return senderEmailData

def populate_signature_response(entities: dict, signature_block: str, json_request_metadata: dict):
    """ populate the extracted entities and signature block in the signaure response format.

    Args:
        entities (dict): list of extracetd entities.
        signature_block (str): the extracted signature block.
        json_request_metadata (dict): meta data of request.

    Returns:
        SignatureResponse: populated signaure respones including the entities.
    """
    
    # if there is not signature block extracted, then return empty response.
    if len(signature_block) == 0:
        return [ SignatureResponse(
            parseStatusType = ParseStatusType.NO_SIGNATURE_BLOCK.value,
            signatureFields = None,
            senderEmailData = None,
            signatureBlock = None ) ]
            
    # populate the entities in a dictionary.    
    fields = {}
    for item in entities:
        if item['entity'] in fields:
            fields[item['entity']].append(item)
        else:
            fields[item['entity']] = [item]
    
    # instantiate a signature block.
    signatureBlock = SignatureBlock( block=signature_block )
    
    # instantiate a status type.
    parseStatusType = ParseStatusType.SUCCESS.value if len( signature_block) > 0 else ParseStatusType.NO_SIGNATURE_BLOCK.value 
    
    # instantiate and populate a signaure fields. 
    signatureFields = SignatureFields(
        
        names = [ Name(
                wholeName="",
                prefix=fields["NAME_PREFIX"][0]['name'] if "NAME_PREFIX" in fields else "",
                given=fields["NAME_GIVEN"][0]['name'] if "NAME_GIVEN" in fields else "",
                middle=fields["NAME_MIDDLE"][0]['name'] if "NAME_MIDDLE" in fields else "",
                surname=fields["NAME_SURNAME"][0]['name'] if "NAME_SURNAME" in fields else "",
                suffix=fields["NAME_SUFFIX"][0]['name'] if "NAME_SUFFIX" in fields else "" ) ],

        jobTitles = populate_entity( JobTitle, "JOB_TITLE", "title", fields ),
        organizations = populate_entity( Organization, "ORGANIZATION", "name", fields ),
        addresses = populate_entity( Address, "ADDRESS_STREET", "line", fields ),
        cities = populate_entity( City, "ADDRESS_CITY", "name", fields ),
        countryDivisions = populate_entity( CountryDivision, "ADDRESS_COUNTRY_DIVISION", "name", fields ),
        countries = populate_entity( Country, "ADDRESS_COUNTRY", "name", fields ),
        phoneNumbers = populate_phones(fields),
        emails = populate_entity( Email, "EMAIL", "address", fields, tag=None, wholeEmail=None ),
        postalCodes = populate_entity( PostalCode, "ADDRESS_POSTAL_CODE", "code", fields ),
        webSites = populate_entity( WebSite, "URL", "url", fields, tag=None, wholeUrl=None )
    )
    
    # generate the metadata objet for senderEmailData.
    if json_request_metadata:
        senderEmailData = generate_response_metadata(json_request_metadata, populate_entity(
                Email, "EMAIL", "address", fields, tag=None, wholeEmail=None))
    else:
        senderEmailData = None
    
    # populate and return the signature response.
    return [SignatureResponse(
            signatureBlock = signatureBlock,
            parseStatusType = parseStatusType,
            signatureFields = signatureFields,
            senderEmailData = senderEmailData
        )]


def load_tokenizer_and_model(model_path):
    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        log.error(
            f"Couldn't find saved pretrained model and tokenizer on path <{model_path}>. Pretrained model is saved by default on './pretrained/<dd.mm.yy-time>'")
        sys.exit(1)
    loaded_model = AutoModelForTokenClassification.from_pretrained(model_path)
    return loaded_tokenizer, loaded_model

def extract_signature(email_text, tokenizer, model, minimum_char_length_between_email_blocks, remove_header=True, cpu_only=False):
    # convert text to UTF-8 to be usable by the `email` package
    inference = {}
    if remove_header:
        email_text = remove_email_header(email_text)
    device = torch.device('cpu') if cpu_only else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    eval_tokenized = tokenizer(email_text,
                               return_tensors="pt",
                               max_length=512,
                               padding='max_length',
                               truncation=True, return_offsets_mapping=True)
    inputs = {
        'input_ids': eval_tokenized['input_ids'].to(device),
        'attention_mask': eval_tokenized['attention_mask'].to(device)
    }

    labels_eval = np.ones(shape=(inputs['input_ids'].size(
        0), inputs['input_ids'].size(1)), dtype=np.int64)
    labels_eval = torch.tensor(labels_eval).to(device)  # Batch size 1

    outputs = model(**inputs, labels=labels_eval)
    logits = outputs.logits
    predictions = np.argmax(logits.cpu().detach().numpy(), axis=2)
    input_ids = eval_tokenized.data["input_ids"]
    # Contains a list of original offsets for tokens within string (list of span tuples, some of them are (0, 0))
    signature_tokenized = [eval_tokenized.encodings[0].offsets[i] for i, _ in enumerate(input_ids[0]) if
                           predictions[0][i] == 1]
    # Remove (0, 0)
    signature_tokenized = list(
        filter(lambda a: a != (0, 0), signature_tokenized))
    if signature_tokenized:
        # Unpacking list here in case of single signature extract
        # signature = email_text[signature_tokenized[0][0]:signature_tokenized[-1][1]]
        signature_pieces = []
        start = signature_tokenized[0][0]
        for i in range(len(signature_tokenized) - 1):

            if (signature_tokenized[i + 1][0] - signature_tokenized[i][1]) > minimum_char_length_between_email_blocks:
                stop = signature_tokenized[i][1]
                signature_pieces.append(
                    (email_text[start:stop], (start, stop)))
                start = signature_tokenized[i + 1][0]

        stop = signature_tokenized[-1][1]
        signature_pieces.append((email_text[start:stop], (start, stop)))
        # piece_lengths = [len(piece[0]) for piece in signature_pieces]

        # max_length_piece = signature_pieces[piece_lengths.index(max(piece_lengths))]
        # return max_length_piece
        inference[SignatureLabels.signature.value] = signature_pieces[-1]
    return inference

def get_signature_request(signature_block: str):
    return {
        "emailMetadata": {
            "toEmailAddresses": [ { "address": "...", "displayName": "..." } ],
            "ccEmailAddresses": [ { "address": "...", "displayName": "..." } ],
            "bccEmailAddresses": [ { "address": "...", "displayName": "..." } ],
            "senderEmailAddress": { "address": "...", "displayName": "..." },
            "subject": "...",
            "messageId": "...",
            "replyToEmailAddresses": [ { "address": "...", "displayName": "..." } ],
            "fromEmailAddresses": [ {"address": "...", "displayName": "..." } ],
            "sentDate": 12345,
            "receivedByEmailAddress": "...",
            "inReplyToMessageIds": [ "...", "..." ],
            "referenceMessageIds": [ "...", "..."]
        },
        "signatureBlock": signature_block
    }

def process_signature(plain_text, text_response):
    """ Postprocessing step in which all signature parts are identified and full lines containing the signature block are selected.

    Args:
        plain_text (str): the original text that was sent to the signature extraction service.
        text_response (str): the output of the signature extraction service.

    Returns:
        tuple: tuple of three blocks: (1) text before signature, (2) signature block, and (3) text after signature
    """

    # if text_response is empty, returns the full email in block 0 and empty blocks 1 and 2
    # if text_response.strip() == '': # REPLACED BY <10 CHARACTERS CONDITION
    # if text_response is less than 10 characters, returns the full email in block 0 and empty blocks 1 and 2
    if not text_response or len(text_response.strip()) < 10:
        return (plain_text, '', '', -1, -1)

    # if text_response is not empty, splits it on blanks (notice first and last character are removed
    # due to some insertions that might occur in the signature extracted block, this issue must be revised!)
    parts = [x for x in text_response.strip()[1:-1].strip().split(' ')
             if x != '']

    # creates a list of offsets for each item in parts, according to the item occurrences in the full
    # email text (plain_text), a valid chunk is an item that can be found within the email body
    idxs = []
    validchunks = []
    for item in parts:
        splits = plain_text.split(item)
        if len(splits) == 1:
            # print('problem 1: a part '+item.replace('\n','|n')+' does not match!')
            # this problem can be ignored for now, but need to investigate insertions/deletions that
            # cause this problem, it might be related to the same problem of insertions addressed above
            continue
        validchunks.append(item)
        offset = len(item)
        chunksizes = [len(splits[0])]
        for chunk in splits[1:-1]:
            chunksizes.append(chunksizes[-1]+offset+len(chunk))
        idxs.append(chunksizes)

    # if the list of offsets results to be an empty list, which means none of the items in parts produced
    # a single valid chunk (i.e. none of the items were found in the email body), returns the full email in
    # block 0 and empty blocks 1 and 2
    if len(idxs) == 0:
        return (plain_text, '', '', -1, -1)

    # looks for a valid chunk that occurs the minimum number of times in the email body, then its
    # corresponding index in idxs list is captured in startidx and its offset value is used to initialize
    # offset variables minoffset and maxoffset
    min_nchunks = min([len(x) for x in idxs])
    if min_nchunks > 1:
        # print('problem 2: all signature chunks appear a minimum of '+str(min_nchunks)+' times!')
        # this might be due to more than one signature occurring, for now, it will use the last one
        for k, offsets in enumerate(idxs):
            if len(offsets) == min_nchunks:
                idxs[k] = [offsets[-1]]
    for k, offsets in enumerate(idxs):
        if len(offsets) == 1:
            startidx = k
            minoffset = offsets[0]
            maxoffset = offsets[0]
            break

    # moves backwards from the selected valid chunk (startidx) to get the signature span preceding it
    for k in range(startidx-1, -1, -1):
        mindif = minoffset-0
        tempmin = max(0, minoffset-1)
        for idx in idxs[k]:
            if idx >= tempmin:
                continue
            tempdif = tempmin-idx
            if tempdif < mindif:
                mindif = tempdif
                minoffset = idx

    # moves forward from the selected valid chunk (startidx) to get the signature span after it
    for k in range(startidx+1, len(idxs)):
        mindif = len(plain_text)-maxoffset
        tempmax = min(len(plain_text), maxoffset+1)
        for idx in idxs[k]:
            if idx <= tempmax:
                continue
            tempdif = idx-tempmax
            if tempdif < mindif:
                mindif = tempdif
                maxoffset = idx

    # uses the expanded signature span (from minoffset to maxoffset) to pull the full signature lines
    maxoffset = maxoffset+len(validchunks[-1])
    signaturechunk = plain_text[minoffset:maxoffset]
    prefix = plain_text[:minoffset].split('\n')[-1]
    suffix = plain_text[maxoffset:].split('\n')[0]

    # constructs the three output blocks: (1) text before signature, (2) signature block (finalchunk),
    # and (3) text after signature
    finalchunk = prefix+signaturechunk+suffix
    beforesignature = '\n'.join(plain_text[:minoffset].split('\n')[:-1])
    aftersignature = '\n'.join(plain_text[maxoffset:].split('\n')[1:])

    return (beforesignature, finalchunk, aftersignature, minoffset, maxoffset)

def ner_parse_response(ner_doc):
    return {f"{SignatureLabels.signature.value}.{item['entity']}": item["name"] for item in ner_doc}
