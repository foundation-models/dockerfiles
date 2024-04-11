import re
import json


def mailtext2json(mailtext):
    cleanedtext = cleanmailtext(mailtext)
    json_message = json.dumps(parsejsonrepmessage(
        cleanedtext.encode('utf-8', errors='ignore').decode('utf-8')))
    return json_message


def cleanmailtext(mailtext):
    cleanedtext = removeheadhtmltag(mailtext)
    cleanedtext = cleanhtml(cleanedtext)
    cleanedtext = removelinks(cleanedtext)
    cleanedtext = removedoblewhitespaces(cleanedtext)
    cleanedtext = removemultiplelinebreaks(cleanedtext)
    return cleanedtext.rstrip()


def cleanhtml(raw_html):
    CLEANR = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub('</p>', "</p>\n", raw_html)
    cleantext = re.sub('<br\s?\/>|<br>', "\n", cleantext)
    cleantext = re.sub(CLEANR, '', cleantext)
    return cleantext


def removelinks(text):
    CLEANR = re.compile(r'\[(http.*?)\]')
    return re.sub(CLEANR, '', text)


def removemultiplelinebreaks(text):
    CLEANR = re.compile(r'[\r\n]{3,}')
    return re.sub(CLEANR, r'\n\n', text)


def removedoblewhitespaces(text):
    CLEANR = re.compile(r'[^\S\r\n]{2,}')
    return re.sub(CLEANR, ' ', text)


def removeheadhtmltag(text):
    CLEANR = re.compile(r'<head>(?:.|\n|\r)+?<\/head>')
    return re.sub(CLEANR, '', text)


def cleanhtmlcomment(raw_html):
    CLEANR = re.compile(r'<!--.*|\r|\n-->')
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


def parsejsonrepmessage(message_text):
    parsed_message = json.loads(json.dumps({"data": {"body": message_text}}))
    return parsed_message

def convert_request(old_data):
    if type(old_data) == list:
        ndarray = [od["emailBody"]["textBody"] for od in old_data]
    else:
        ndarray = [old_data["emailBody"]["textBody"]]

    new_data = {"data": {"names": ["payload"], "ndarray": ndarray}}
    return new_data

def convert_request_body(old_data):
    if type(old_data) == list:
        ndarray = [od["emailBody"]["textBody"] for od in old_data]
    else:
        ndarray = [old_data["emailBody"]["textBody"]]

    new_data = {"data": {"names": ["payload"], "ndarray": ndarray}}
    return new_data

def convert_request_signature(data):
    if data:
        value = data["signatureBlock"]
    else :
        return ''
    return value

#----- Added code --------#
def convert_request_email_metadata(data):
    return data["emailMetadata"] if data else ''
#----- Added code --------#


def parse_text(input_text: str, json_data: dict, language_keys: dict):
    """Parse the json file to a text prompt format."""

    task_label = language_keys['TASK']
    text_label = language_keys['TEXT']
    examples_label = language_keys['EXAMPLES']

    # Get the task
    task = json_data['TASK']
    # Get the fields
    fields = json_data['FIELDS']
    # Get the separator
    # separator = json_data['SEPARATOR']
    # Get the examples
    examples = json_data['EXAMPLES']
    # Get the input text
    # input_text = json_data['INPUT_TEXT']

    # Create the prompt question
    prompt_task = f"{task_label}: {task} \n\n"

    # Create the prompt separator
    # prompt_separator = separator + "\n\n"

    prompt_response_head_format =""

    # Create the prompt examples
    prompt_examples = f"{examples_label}: \n\n"
    for _, example in examples.items():
        prompt_examples += f"{text_label}: {example['TEXT']} \n\n"
        if fields and len(fields) > 0:
            for field in fields:
                #print(f"[{field}]: {example['VALUES']}\n")
                prompt_examples += f"[{field}]: {example['VALUES'].get(field,'')}\n"
        else:
           prompt_examples +=  f"{example['VALUES']}\n"
        # prompt_examples += "\n" + prompt_separator

    # Create the response format head
    #prompt_response_head_format += f"{text_label}: " + input_text + "\n\n"
    prompt_response_head_format += f"{text_label}: {input_text}"

    # Create the prompt
    prompt = prompt_task + prompt_examples + prompt_response_head_format

    return prompt

def get_prompt_keys_by_language(language):
    if language == "en" or language == None:
        language_dict =  {"TEXT": "TEXT",
                          'TASK': 'TASK',
                          'EXAMPLES': 'EXAMPLES'}

    if language == "zhs":
        language_dict =  {"TEXT": "文本",
                          'TASK': '任务',
                          'EXAMPLES': '例子'}

    return language_dict