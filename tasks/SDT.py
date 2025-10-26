###############################################################################
# Imports
import os
import re
import pandas as pd
import random

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.utils import FEW_SHOT_PROMPT_TEMPLATE, process_responses_extraction

###############################################################################

###############################################################################
# Constants
LABEL2ID = {
    "EMPTY": -1,  # Special label for empty responses
    "O": 0,
    "SIM": 1,
}
LABELS = list(LABEL2ID.keys())


# Columns to be logged together with the model's raw responses
# Note: 'txt_src' is used in DharmaBench format, 'filename' in legacy format
LOG_COLUMNS = ["id", "text", "ground_truth"]

###############################################################################


###############################################################################
# Prompts
# PROMPTS = {
#     "system": """You are a computational linguist and philologist specializing in identifying similes expressions in Classical Tibetan texts.

# Your task is to analyze a given sentence or verse written in Tibetan and extract simile expressions it contains.

# Definitions:
# - A simile: an explicit comparison between two entities, typically marked by comparison words such as "lta bu", "bzhin", "lta", "’dra", etc.

# Annotation Guideline:
# - Identify all minimal spans that contain a simile.
# - Label each identified span as "SIM" (simile).
# - Do not annotate literal or descriptive phrases that do not rely on identification or comparison.

# Return your output as a JSON with "prediction" key. The value is a list of tuple (LABEL, SPAN), each with:
# - LABEL: "SIM"
# - SPAN: the exact text span that contains the simile (minimal required span)

# Example for an item: ("SIM", "de bzhin du")

# If no simile is found, return an empty list under the "prediction" key.
# Only respond with the JSON output, do not include any additional text or explanations.
# """,
#     "user": "Text: {text}\n"
# }

PROMPTS = {
    "system": """You are a computational linguist and philologist specializing in identifying similes expressions in Classical Tibetan texts.

Your task is to analyze a given sentence or verse written in Tibetan and extract simile expressions it contains.

Definitions:
- A simile: an explicit comparison between two entities, typically marked by comparison words such as "ལྟ་བུ", "བཞིན", "ལྟ", "འདྲ", etc.

Annotation Guideline:
- Identify all spans containing a simile.
- Label each identified span as "SIM" (simile).
- Do not annotate literal or descriptive phrases that do not rely on identification or comparison.
- Mark the absolute minimal span that contains the simile, even if it is part of a larger phrase.

Return your output as a JSON with "prediction" key. The value is a list of dictionaries, each with:
- LABEL: "SIM"
- SPAN: the exact text span that contains the simile (minimal required span)

Example for an item: {{"LABEL": "SIM", "SPAN": "དེ་བཞིན་དུ"}}

If no simile is found, return an empty list under the "prediction" key.
Only respond with the JSON output, do not include any additional text or explanations.
""",
    "user": "Text: {text}\n"
}


# FEW_SHOT_EXAMPLES = [
#     {
#         "input": "she'i thugs gang gi chu gter 'dra",
#         "output": {
#             "prediction": [
#                 ["SIM", "'dra"]
#             ]
#         }
#     },
#     {
#         "input": "sgam po lta bu'i blo gros la brten nas",
#         "output": {
#             "prediction": [
#                 ["SIM", "lta bu"]
#             ]
#         }
#     },
#     {
#         "input": "thugs rje'i chu bo la sogs pa'i snying rje sprin ltar 'phur",
#         "output": {
#             "prediction": [
#                 ["SIM", "sprin ltar"]
#             ]
#         }
#     },
#     {
#         "input": "bdag cag bod du byon pa'i lo rgyus mdor bsdus nas brjod do",
#         "output": {
#             "prediction": []
#         }
#     }
# ]


###############################################################################


###############################################################################
# Functions

def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(data_dir, "Tibetan/SDT/test.jsonl")
    if os.path.exists(path):
        data = pd.read_json(path, lines=True)

    data["ground_truth"] = data["ground_truth"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])
    data["offsets"] = data["offsets"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])    
    
    return None, data


def _get_few_shot_prompt(
    seed: int = 42,
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param seed: seed for reproducibility
    :param lang: language, either "sanskrit" or "tibetan"
    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility
    examples = FEW_SHOT_EXAMPLES
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

    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = _get_few_shot_prompt(seed=seed)
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
        {"text": "".join(row["text"])}
        for _, row in data.iterrows()
    ]
    return user_inputs


process_responses = process_responses_extraction

###############################################################################

###############################################################################
# Export
SDT = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}
