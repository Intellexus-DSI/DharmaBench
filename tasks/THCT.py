###############################################################################
# Imports
import os
import re
import pandas as pd
import random
import json

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.utils import FEW_SHOT_PROMPT_TEMPLATE, process_responses_classification, process_responses_extraction

###############################################################################

###############################################################################
# Constants
LABEL2ID = {
    "EMPTY":                         -1,
    "Vinaya":                         0,
    "Sūtra":                          1,
    "Tantra":                         2,
    "Dhāraṇī":                        3,
    "Non-tantric eulogies":           4,
    "Treatises on Tantric Topics":    5,
    "Treatises on Sūtras":            6,
    "Treatises on Madhyamaka":        7,
    "Treatises on Cittamātra":        8,
    "Treatises on Abhidharma":        9,
    "Treatises on Vinaya":           10,
    "Jātaka tales":                  11,
    "Epistles":                      12,
    "Treatises on Pramāṇā":          13,
    "Treatises on Sanskrit":         14,
    "Treatises on medicine":         15,
    "Treatises on arts and crafts":  16,
}
LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["text", "label"]

###############################################################################


###############################################################################
# Prompts


"""You are a computational linguist, philologist, and scholar of Sanskrit textual traditions. 
    Your task is to analyze a given root-text and a given commentary written in Sanskrit and decide if the commentary is indeed a commentary of the root.
    Classification criteria:
    - TRUE: The commentary is relevant and provides insight into the root-text.
    - FALSE: The commentary is not relevant or does not provide insight into the root-text.

    Response Guideline:
    Return your output as a JSON with a "label" key. The value is either "TRUE" or "FALSE". For example
    - "label": "TRUE"
    or
    - "label": "FALSE"
    Only respond with the JSON output, do not include additional text or explanations.
    """,

PROMPTS = {
    "system": """You are a computational linguist, philologist, and scholar of Sanskrit and Tibetan textual traditions, specializing in the classification of canonical and paracanonical works by genre.

Your task is to analyze a given passage written in Tibetan and classify it to exactly one of these established categories:

    - Vinaya
    - Sūtra
    - Tantra
    - Dhāraṇī
    - Non-tantric eulogies
    - Treatises on Tantric Topics
    - Treatises on Sūtras
    - Treatises on Madhyamaka
    - Treatises on Cittamātra
    - Treatises on Abhidharma
    - Treatises on Vinaya
    - Jātaka tales
    - Epistles
    - Treatises on Pramāṇā
    - Treatises on Sanskrit
    - Treatises on medicine
    - Treatises on arts and crafts

    Response Guideline:
    - Return your output as a JSON with a "label" key. The value is one of the categories above. For example:
    - {{"label": "Non-tantric eulogies"}}
    or
    - {{"label": "Epistles"}}

    Only respond with the JSON output, do not include additional text or explanations. do not include any additional text or explanations. 
""", 
    "user": "Text: {text}\n"
}


###############################################################################


###############################################################################
# Functions

def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(data_dir, "Tibetan/THCT/test.jsonl")
    if os.path.exists(path):
        test = pd.read_json(path, lines=True)
        if 'class' in test.columns and 'label' not in test.columns:
            test['label'] = test['class']
        train = None

    return train, test


def get_prompt(config: dict, **kwargs) -> ChatPromptTemplate:
    """
    Get the prompt for the given task and prompt type.
    Also returns the schema for the task if model supports structured output.
    :param config: configuration
    :param kwargs: keyword arguments
    :return: prompt
    """
    # Get kwargs
    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt


def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs as a list of dictionaries
    """
    return [{"text": row["text"]} for _, row in data.iterrows()]

process_responses = process_responses_classification

# The evaluation mode  for extraction tasks ONLY
EVAL_MODE = None

###############################################################################

###############################################################################
# Export
THCT = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
    "eval_mode": EVAL_MODE
}