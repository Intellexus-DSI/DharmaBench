###############################################################################
# Imports
import os
import re
import pandas as pd
import numpy as np
import random
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from matplotlib.cbook import get_sample_data

from utils import utils
from utils.utils import FEW_SHOT_PROMPT_TEMPLATE, process_responses_classification, process_responses_extraction

from lxml import etree # added for this task to handle xml

###############################################################################

###############################################################################
# Constants
LABEL2ID = {"quote": 1, "author": 2, "title": 3}
LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(parent_dir, "data/sanskrit/citation")


# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["id", "text", "ground_truth"]

###############################################################################


###############################################################################
# Prompts
FEW_SHOT_EXAMPLES = {
    "english": {
        "cot": [
            {
                "input": "sentence: The very solicitors' boys who have kept the wretched suitors at bay, by protesting time out of mind that Mr Chizzle, Mizzle, or otherwise was particularly engaged and had appointments until dinner, may have got an extra moral twist and shuffle into themselves out of Jarndyce and Jarndyce.",
                "output": """sentence: The very solicitors' boys who have kept the wretched suitors at bay, by protesting time out of mind that Mr Chizzle, Mizzle, or otherwise was particularly engaged and had appointments until dinner, may have got an extra moral twist and shuffle into themselves out of Jarndyce and Jarndyce.
explanation: Time out of mind" is idiomatic, meaning for an extremely long time rather than a specific duration. Here, it emphasizes the endless nature of the legal case.
idioms: ['time out of mind']""",
            },
            {
                "input": "sentence: In 1746, some months after his 36th birthday, Samuel Johnson, that great literary figure of the 18th century, affectionately referred to as the Good Doctor, began work on his monumental Dictionary of the English Language.",
                "output": """sentence: In 1746, some months after his 36th birthday, Samuel Johnson, that great literary figure of the 18th century, affectionately referred to as the Good Doctor, began work on his monumental Dictionary of the English Language.
explanation: The Good Doctor" is idiomatic because it serves as an affectionate, honorific nickname rather than a literal description of Johnson’s medical expertise. It conveys respect and familiarity, commonly used for well-regarded scholars or physicians.
idioms: ['the Good Doctor']""",
            },
            {
                "input": "sentence: The film briefly visits a “rubber room” in New York City where idle teachers accused of misconduct wait months and sometimes years for hearings while drawing full salaries.",
                "output": """sentence: The film briefly visits a “rubber room” in New York City where idle teachers accused of misconduct wait months and sometimes years for hearings while drawing full salaries.
explanation: Rubber room" is idiomatic because it does not refer to a room made of rubber but rather a slang term for a detention-like space where teachers accused of misconduct are sent while awaiting hearings. The phrase carries a figurative meaning, implying a bureaucratic limbo or an absurd, almost surreal situation.
idioms: ['rubber room']""",
            },
            {
                "input": "sentence: We have no decisions in our state directly on point. With us the problem is one of first impression. None of the cases cited is on point.",
                "output": """sentence: We have no decisions in our state directly on point. With us the problem is one of first impression. None of the cases cited is on point.
explanation: There are no idioms in this sentence because "on point" and "one of first impression" are legal terms used in their precise, technical sense rather than figuratively. They retain their literal legal meanings.
idioms: []""",
            },
            {
                "input": "sentence: To find the real Michel Foucault is to ask “which one”?, Should we look at the life of the man himself, who as a boy wanted to be a goldfish, but became a philosopher and historian, political activist, leather queen, bestseller, tireless campaigner for dissident causes?",
                "output": """sentence: To find the real Michel Foucault is to ask “which one”?, Should we look at the life of the man himself, who as a boy wanted to be a goldfish, but became a philosopher and historian, political activist, leather queen, bestseller, tireless campaigner for dissident causes?
explanation: There are no idioms in this sentence because all phrases, including "wanted to be a goldfish" and "leather queen," are used either literally, descriptively, or as personal characterizations rather than established idiomatic expressions with non-literal meanings.
idioms: []""",
            },
            {
                "input": "sentence: Compared to other modern languages, C is much less forgiving, much more terse, and generally much more ill tempered. However, it is about as close to programming on the bare metal as you can get while still using a well-supported language with good library support for everything from numeric calculations to graphics.",
                "output": """sentence: Compared to other modern languages, C is much less forgiving, much more terse, and generally much more ill tempered. However, it is about as close to programming on the bare metal as you can get while still using a well-supported language with good library support for everything from numeric calculations to graphics.
explanation: There are no idioms in this sentence because "on the bare metal" is a common technical phrase in computing, used in its literal sense within the field. It describes programming close to hardware rather than serving a metaphorical or figurative purpose.
idioms: []""",
            },
        ],
    }
}


PROMPTS = {
    "system": """You are a classic Sanskrit philologist and linguist specializing in finding citations and authors written in IAST Sanskrit. 

Your task is to analyze a given sentence or verse written in Sanskrit and extract any Quote, Author, Title it contains.

Definitions:
A Quote: a short sentence or paragraph citation of another work written in Sanskrit.
An Author: the name of the author that is being cited in a work written in Sanskrit.
A Title: the title of the other work written in Sanskrit.

Annotation Guideline:
- A sample might have 0, 1 or even multiple citations.
- Think carefully about cultural and topical relevance in classic Sanskrit. 
- If detected, write the Quote, Author, Title exactly as they are in the sentence, without any changes. Only answer in JSON.

Return your output as a JSON with "prediction" key. The value is a list of dictionaries, each with:
- LABEL: either "Quote" or "Author" or "Title".
- SPAN: the exact text span that contains the "Quote" or "Author" or "Title" (minimal required span), exactly as it appears in the input text.

If no "Quote" or "Author" or "Title" is found, return an empty list under the "prediction" key.
Only respond with the JSON output, do not include any additional text or explanations.
""",

    "user": "Text: {text}\n"
}



###############################################################################


###############################################################################
# Functions
"""
If there is one input text, name it 'text' in the dataframe.
If it's a classification task, df should contain 'label' column with the labels as strings.
If it's an extraction task, df should contain 'ground_truth' column with a list as follows:
---------- [("PER", "John Doe"), ("LOC", "New York")...] ----------
"""
def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = os.path.join(data_dir, "Sanskrit/QUDS/test.jsonl")
    train = os.path.join(data_dir, "Sanskrit/QUDS/train.jsonl")
    
    if os.path.exists(test):
        test = pd.read_json(test, lines=True)
        if os.path.exists(train):
            train = pd.read_json(train, lines=True)
        else:
            print("Dataframe empty")
        
        if 'text' in test.columns and 'sample' not in test.columns:
            test['sample'] = test['text']
        if not train.empty and 'text' in train.columns and 'sample' not in train.columns:
            train['sample'] = train['text']

    return train, test

def get_path_data(path):
    files = os.listdir(path)
    df = pd.DataFrame()

    for file in files:
        filename = os.path.join(path, file)

        new_loaded_df = pd.read_csv(filename)
        df = pd.concat([df, new_loaded_df])

    df["ground_truth"] = df["ground_truth"].apply(lambda x: eval(x))
    df["sample"] = df["text"].apply(lambda x: x)

    return df

def get_text_data(file_name, text):
    extracted_samples = []

    samples = re.split("---", text)

    for sample_no, sample in enumerate(samples):
        sample = sample.strip()
        if sample != "":
            single_sample_data = get_single_sample_data(file_name, sample_no, sample)
            if single_sample_data is not None:
                extracted_samples.append(single_sample_data)

    return extracted_samples

def get_single_sample_data(file_name, sample_no, text):

    #print(f'Processing file_name [{file_name}], sample [{sample_no}]')
    root = etree.fromstring(f'<root>{text}</root>')

    authors = extract_specific_tag_from_etree(root, 'author')
    titles = extract_specific_tag_from_etree(root, 'title')
    quotes = clean_tag_texts(extract_specific_tag_from_etree(root, 'quote'))

    clean_text = clean_text_string(etree.tostring(root, encoding='unicode', method='text').replace("\t", " "))
    sample_id = get_sample_id(file_name, sample_no)

    ground_truth = []
    for author in authors:
        ground_truth.append(("Author", author))
    for title in titles:
        ground_truth.append(("Title", title))
    for quote in quotes:
        ground_truth.append(("Quote", quote))

    if len(ground_truth) == 0:
        #print(f'No ground truth found for sample [{sample_id}] :')
        #print(f'')
        #print(f'{clean_text}')
        #print(f'')
        return None

    return {"sample_id": sample_id, "text": clean_text, "ground_truth": ground_truth}

def extract_specific_tag_from_etree(root, tag_name: str) -> list:
    texts = root.xpath(f'//{tag_name}/text()')
    return texts

def get_sample_id(file_name, sample_no):
    return f"{file_name}_{sample_no}"

def clean_tag_texts(texts):
    for i, text in enumerate(texts):
        texts[i] = clean_text_string(text)
    return texts

def clean_text_string(text):
    return text.replace("\t", " ").strip()

def _get_few_shot_prompt(
    shots: int = 6, seed: int = 42
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param shots: number of few-shot examples
    :param seed: seed for reproducibility
    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility

    ########################################
    # Get data
    random.seed(seed)
    train, _ = get_data()
    sample_df = train.sample(min(shots, len(train)), random_state=seed)
    examples = [
        {
            "input":  f"Text: {row['text']}",
            "output": {"label": row["ground_truth"]}
        }
        for _, row in sample_df.iterrows()
    ]
    ########################################
    # Shuffle the examples
    random.shuffle(examples)  # Shuffle the list in place

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=examples,
    )
    return few_shot_prompt

def get_prompt(config: dict, **kwargs) -> ChatPromptTemplate:
    """
    Get the prompt for the given task and prompt type.
    Also returns the schema for the task if model supports structured output.
    :param config: configuration
    :param kwargs: keyword arguments
    :return: prompt
    """
    # Get kwargs
    prompt_type = config["prompt_type"]
    seed = config["seed"]
    shots = config["shots"]

    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = _get_few_shot_prompt(shots=shots, seed=seed)
        # Add in position 1 (after the system prompt)
        messages.insert(1, few_shot_prompt)
    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt

def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs as a list of dictionaries
    """
    user_inputs = [
        {"text": row["text"]}
        for _, row in data.iterrows()
    ]
    return user_inputs

def process_sanskrit_citation_responses_extraction(
    responses: list[dict],
    test: pd.DataFrame,
    pred_col: str = "predictions",
    true_col: str = "ground_truth",
    text_col: str = "text",
    mode: str = "type",
    **kwargs
) -> tuple[dict, pd.DataFrame]:
    """
    Process extraction responses (span-based), calculate MUC metrics, and augment the test DataFrame.

    :param responses: List of model outputs, each as a dict with 'prediction' key holding extracted spans.
    :param test: The original test DataFrame containing the ground truths and text.
    :param pred_col: Column name for predictions (to be filled).
    :param true_col: Column name for true spans (ground truth).
    :param text_col: Column name for input text.
    :param mode: Evaluation mode for MUC metrics ('type', 'exact', 'partial', 'strict')
    :return: Tuple of (metrics dict, augmented DataFrame)
    """

    predictions = []
    hallucinated = []

    for resp in responses:
        response_predictions = []
        # print(f"resp: {resp}")

        try:
            if resp and "responses" in resp:
                responses = resp["responses"]

                for response in responses:
                    if "response" in response:
                        response = response["response"]

                        if isinstance(response, list):
                            for citation in response:
                                add_citation_predictions(response_predictions, citation)
                        elif isinstance(response, dict):
                            add_citation_predictions(response_predictions, citation)
                        else:
                            print(f"Unsupported response type:[{type(response)}] for response:[{response}]")


            if len(response_predictions) == 0:
                hallucinated.append(True)
            else:
                hallucinated.append(False)

        except Exception as e:
            print(f"Error parsing prediction: {e} → {resp}")
            response_predictions = []
            hallucinated.append(True)

        predictions.append(response_predictions)

    # Normalize ground truths and texts
    # ground_truth = []
    # for _, row in test.iterrows():
    #     row_ground_truth = []
    #     values = row[true_col]
    #     print(f"values: {values}")
    #     print(f"values len: {len(values)}")
    #     for new_value in values:
    #         print(f"new_value: {new_value}")
    #         tag, span = new_value
    #         normalized_span = span.lower().replace("-", "")
    #         row_ground_truth.append((tag, normalized_span))
    #     ground_truth.append(row_ground_truth)

    # Normalize ground truths and texts
    ground_truth = [
        [(tag, span.lower().replace("-", "")) for tag, span in row[true_col]]
        for _, row in test.iterrows()
    ]

    # texts = []
    # for _, row in test.iterrows():
    #     print(f"row: {row}")
    #     print(f"text_col: {text_col}")
    #     row_text = row[text_col].lower().replace("-", "")
    #     texts.append(row_text)

    texts = [row[text_col].lower().replace("-", "") for _, row in test.iterrows()]

    print(f"predictions: {predictions}")
    #print(f"ground_truth: {ground_truth}")

    # Compute metrics
    csv_results, metrics = utils.calc_metrics_extraction(ground_truth, predictions, texts, verbose=False)
    metrics["hallucinations"] = int(sum(hallucinated))
    csv_results["hallucinations"] = metrics["hallucinations"]

    # Add predictions and hallucination flags to DataFrame
    test[pred_col] = predictions
    test["hallucinated"] = hallucinated

    return csv_results, metrics, test

def add_citation_predictions(preds, citation):
    add_citation_prediction(preds, citation, "Quote")
    add_citation_prediction(preds, citation, "Author")
    add_citation_prediction(preds, citation, "Title")

def add_citation_prediction(preds, citation, citation_type):
    if citation_type in citation:
        value = citation[citation_type]
        if value is not None:
            if value != "" and value != "Unknown":
                clean_pred = clean_prediction_text(value)
                pred = (citation_type, clean_pred)
                preds.append(pred)

def clean_prediction_text(text):
    text = text.replace('<span class="quote">', '')
    text = text.replace('</span>', '')
    text = text.replace('<span class="author">', '')
    return text.strip()

process_responses = process_responses_extraction # process_sanskrit_citation_responses_extraction # process_responses_extraction #process_responses_classification


###############################################################################

###############################################################################
# Export
QUDS = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}


