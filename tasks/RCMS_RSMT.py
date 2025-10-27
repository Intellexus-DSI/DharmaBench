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
LABEL2ID = {"EMPTY": -1, "TRUE": 1, "FALSE": 0}
LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(parent_dir, "data/rcc")


# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["pair_id", "root", "commentary", "label"]

###############################################################################

###############################################################################
# Helper Functions
def _get_language(task: str) -> str:
    """
    Convert task initials to formal language name.
    :param task: task name containing language initials
    :return: formal language name
    """
    if "RCMS" in task:
        return "sanskrit"
    elif "RCMT" in task:
        return "tibetan"
    else:
        raise ValueError("Invalid task specified. Use 'skt' or 'tib' in the task name.")

###############################################################################

###############################################################################
# Prompts
PROMPTS = {
    "system_skt": """You are a computational linguist, philologist, and scholar of Sanskrit textual traditions. 
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

    "system_tib": """You are a computational linguist, philologist, and scholar of Classical Tibetan textual traditions. 
    Your task is to analyze a given root-text and a given commentary written in Tibetan and decide if the commentary is indeed a commentary of the root.
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
    "user": "Root:{root}\nCommentary:{commentary}\n",
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
    task = kwargs.get("task", "")
    lang = _get_language(task)
    
    if lang == "sanskrit":
        test = os.path.join(data_dir, "Sanskrit/RCMS/test.jsonl")
        train = os.path.join(data_dir, "Sanskrit/RCMS/train.jsonl")
    elif lang == "tibetan":
        test = os.path.join(data_dir, "Tibetan/RCMT/test.jsonl")
        train = os.path.join(data_dir, "Tibetan/RCMT/train.jsonl")
    else:
        print("Lang not found")
    
    if test and os.path.exists(test):
        test = pd.read_json(test, lines=True)
        if os.path.exists(train):
            train = pd.read_json(train, lines=True)
        else:
            print("Train dataframe empty")
        
        if 'id' in test.columns and 'pair_id' not in test.columns:
            test['pair_id'] = test['id']
            if not train.empty:
                train['pair_id'] = train['id']

    return train, test



def _get_few_shot_prompt(
    shots: int = 6, seed: int = 42, lang: str = "sanskrit"
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param shots: number of few-shot examples
    :param seed: seed for reproducibility
    :param lang: language, either "sanskrit" or "tibetan"
    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility

    if lang == "sanskrit":
        # Sanskrit examples
        examples = [
            {
                "input": "Root:√kṛ\nCommentary:karoti - he does/makes",
                "output": {"label": "TRUE"},
            },
            {
                "input": "Root:√gam\nCommentary:gacchati - he goes",
                "output": {"label": "TRUE"},
            },
            {
                "input": "Root:√kṛ\nCommentary:asti - he is",
                "output": {"label": "FALSE"},
            },
            {
                "input": "Root:√bhū\nCommentary:bhavati - he becomes/exists",
                "output": {"label": "TRUE"},
            },
        ]
    else:
        # Tibetan examples
        examples = [
            {
                "input": "Root:byed\nCommentary:byed pa - to do/make",
                "output": {"label": "TRUE"},
            },
            {
                "input": "Root:'gro\nCommentary:'gro ba - to go",
                "output": {"label": "TRUE"},
            },
            {
                "input": "Root:byed\nCommentary:yin - to be",
                "output": {"label": "FALSE"},
            },
            {
                "input": "Root:'dug\nCommentary:'dug pa - to sit",
                "output": {"label": "TRUE"},
            },
        ]

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
    lang = _get_language(config["task"])
    # Get kwargs
    prompt_type = config["prompt_type"]
    seed = config["seed"]
    shots = config["shots"]

    if lang == "sanskrit":
        system_prompt = PROMPTS["system_skt"]
    else:
        system_prompt = PROMPTS["system_tib"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = _get_few_shot_prompt(shots=shots, seed=seed, lang=lang)
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
        {"root": str(row["root"]), "commentary": str(row["commentary"])}
        for _, row in data.iterrows()
    ]
    return user_inputs

process_responses = process_responses_classification

EVAL_MODE = None

###############################################################################

###############################################################################
# Export
RCMS_RSMT = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
    "eval_mode": EVAL_MODE
}
