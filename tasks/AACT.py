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
LABEL2ID = {
    "EMPTY":        -1,  # Special label for empty responses
    "ALLO":          0,
    "AUTO":          1
}
LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["text", "label"]

###############################################################################


###############################################################################
# Prompts
PROMPTS = {
  "system": """You are a computational linguist, philologist, and scholar of Classical Tibetan and Classical Sanskrit textual traditions.

        Your task is to analyze a given passage written in Tibetan and classify its origin category, either Allochthonous or Autochthonous, based on its predominant textual structure.

        Classification criteria:
        - ALLO (Allochthonous): text imported from an external tradition (mainly Sanskrit)
        - AUTO (Autochthonous): native Tibetan material

        Response Guideline:
        Return your output as a JSON with a "label" key. The value is either "ALLO" or "AUTO". For example
        - "label": "ALLO"
        or
        - "label": "AUTO"
        Only respond with the JSON output, do not include additional text or explanations.""",
    "user": "Text: {text}\n"
}

###############################################################################


###############################################################################

def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:

    path = os.path.join(data_dir, "Tibetan/AACT/test.jsonl")
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

###############################################################################

###############################################################################
# Export
AACT = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}