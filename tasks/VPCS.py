###############################################################################
# Imports
import os
import re
import pandas as pd
import random

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.utils import FEW_SHOT_PROMPT_TEMPLATE, process_responses_classification, process_responses_extraction

###############################################################################

###############################################################################
# Constants
# LABEL2ID = {"other": 0, "anuṣṭubh": 1, "anustubh (even pāda)": 2, "anustubh (odd pāda)": 3, "upajāti": 4, "indravajrā": 5, "āryā": 6, "śālinī": 7, "rathoddhatā": 8, "śārdūlavikrīḍita": 9, "sragdharā": 10, "unknown": 11, "false verse": 12}
LABEL2ID = {"verse": 0, "prose": 1, "EMPTY": 2}
LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(parent_dir, "data")
FILE_NAME = "verse_prose.csv"


# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["text", "label"]

###############################################################################


###############################################################################
# Prompts

# FEW_SHOT_EXAMPLES_SKT = [
#         {
#             "input": "paramārthaśāstrakṛtyā kurvāṇaṃ śāstṛkṛtyam iva loke | yaṃ buddhimatām agryaṃ dvitīyam iva buddham ity āhuḥ ||",
#             "output": {
#                 "label": "āryā"
#             }
#         },

# ]

# PROMPTS = {
#     "system": """You are a professional Sanskrit literary analyst specializing in textual classification. Your task is to classify whether a given Sanskrit text (in IAST) is verse or prose.
# Classify the text into one of these labels:
# - verse: Sanskrit text with metrical patterns, rhythmic structure, or chandas.
# - prose: Sanskrit text in ordinary language without metrical constraints.
# - unknown: Classification cannot be determined

# Analyze metrical patterns, line structure, and rhythmic elements in the Sanskrit text.
# IMPORTANT: You must use ONLY these exact 3 labels above. Do not create any new labels or variations.
# Only answer with the label in JSON format with one field "label" and the value being the label name.
#     """,
#     "user": "Text: {text}\n",
# }

PROMPTS = {
    "system": """You are a Sanskrit literary analyst specializing in verse/prose classification.

Classify the given Sanskrit text (IAST) into one of these labels:
- "verse": Text with metrical patterns, rhythmic structure, or chandas
- "prose": Text in ordinary language without metrical constraints  
- "EMPTY": Classification cannot be determined

Analyze for: metrical patterns, line structure, and rhythmic elements.

IMPORTANT: Use ONLY these exact 3 labels. Do not create variations.

Return the classification as 'label' that fits the most to the input text according to its structure.
Only answer with the label, in JSON format with one field "label" and the value being the EXACT label name from the list above.
    """,
    "user": "Text: {text}\n",
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
    path = os.path.join(data_dir, "Sanskrit/VPCS/test.jsonl")
    if os.path.exists(path):
        df = pd.read_json(path, lines=True)
        if 'class' in df.columns:
            df['label'] = df['class'].str.lower()
        elif 'label' in df.columns and df['label'].dtype == 'object':
            df['label'] = df['label'].str.lower()

    # verify there are columns 'text' and 'label', if their sentiment changes it to label
    if "text" not in df.columns:
        raise ValueError("Dataframe must contain 'text' column.")
    if "label" not in df.columns:
        if "sentiment" in df.columns:
            df.rename(columns={"sentiment": "label"}, inplace=True)
            print("Renamed 'sentiment' column to 'label'.")
        elif "class" in df.columns:
            df.rename(columns={"class": "label"}, inplace=True)
            df['label'] = df['label'].str.lower()
            print("Renamed 'class' column to 'label'.")
        else:
            raise ValueError("Dataframe must contain 'label', 'class', or 'sentiment' column.")


    is_test=True
    if is_test:
        # split the dataframe into train and test sets randomly
        # train = df.sample(frac=0.8, random_state=42)  # 80% for training
        # test = df.drop(train.index)  # remaining 20% for testing
        test = df.copy()
        train = pd.DataFrame(columns=df.columns)
    else:
        train = df.copy()
        test = pd.DataFrame(columns=df.columns)

    return train, test

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

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=FEW_SHOT_EXAMPLES,
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
    print(f'Created prompt {prompt}, shots: {shots}, seed: {seed}')
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

process_responses = process_responses_classification

# If entitiy type matters and model predicts it, use 'type'
# If entity type does not matter, and/or model doesn't predict it, use 'partial'
EVAL_MODE = None

###############################################################################

###############################################################################
# Export
VPCS = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
    "eval_mode": EVAL_MODE
}
