###############################################################################
# Imports
import os
import pandas as pd

from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append("../utils/") 
print(sys.path)
from utils.utils import process_responses_extraction, get_few_shot_prompt

###############################################################################

###############################################################################
# Constants
LABEL2ID = {
    "EMPTY": -1,  # Special label for empty responses
    "O": 0,
    "MET": 1,
    "SIM": 2,
}
LABELS = list(LABEL2ID.keys())


# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["id", "text", "ground_truth"]

###############################################################################


###############################################################################
# Prompts
PROMPTS = {
    "system": """
You are a computational linguist and philologist specializing in identifying metaphorical and simile expressions in Classical Sanskrit texts.
Your task is to analyze a given sentence or verse written in Sanskrit and extract any metaphorical or simile expressions it contains.

Definitions:
- A metaphor (rūpaka): a rhetorical figure where the referent (upameya) is identified with a metaphorical form (upamāna) without using an explicit comparison marker such as "iva". These often appear as fused compounds or tightly linked noun phrases (e.g., "cic-candra-candrikā").
- A simile: an explicit comparison between two entities, typically marked by comparative particles like "iva", "yathā", "sadṛśam", etc.

Annotation Guideline:
- Identify all minimal spans that contain a metaphor or simile.
- Label each identified span as either "MET" (metaphor) or "SIM" (simile), according to its rhetorical function.
- Do not annotate literal or descriptive phrases that do not rely on identification or comparison.

Return your output as a JSON with "prediction" key. The value is a list of dictionaries, each with:
- LABEL: either "MET" or "SIM"
- SPAN: the exact text span that contains the metaphor or simile (minimal required span), exactly as it appears in the input text.

If no metaphor or simile is found, return an empty list under the "prediction" key.
Only respond with the JSON output, do not include any additional text or explanations.
""",
    "user": "Text: {text}\n"
}

# Return your output as a JSON with "prediction" key. The value is a list of lists [LABEL, SPAN], each with:
# - LABEL: either "MET" or "SIM"
# - SPAN: the exact text span that contains the metaphor or simile (minimal required span), exactly as it appears in the input text.
#
# If no metaphor or simile is found, return an empty list under the "prediction" key.
# Only respond with the JSON output, do not include any additional text or explanations.
# """,
#
#     "user": "Text: {text}\n"
# }


FEW_SHOT_EXAMPLES = [
    {
        "input": "sā nayanābja-saroruha-śobhāṁ vahati",
        "output": {
            "prediction": [
                ["MET", "nayanābja-saroruha-śobhām"]
            ]
        }
    },
    {
        "input": "sa sūrya iva tejomayaḥ babhau",
        "output": {
            "prediction": [
                ["SIM", "sūrya iva"]
            ]
        }
    },
    {
        "input": "tava vacanam amṛta-dhārā iva sītalaṁ spṛśati manaḥ",
        "output": {
            "prediction": [
                ["MET", "amṛta-dhārā"],
                ["SIM", "iva sītalaṁ"]
            ]
        }
    },
    {
        "input": "rāmaḥ vanaṁ gacchati saha sītayā",
        "output": {
            "prediction": []
        }
    }
]


###############################################################################


###############################################################################
# Functions
def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.join(data_dir, "Sanskrit/SMDS/test.jsonl")
    if os.path.exists(path):
        data = pd.read_json(path, lines=True)

    data["ground_truth"] = data["ground_truth"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])
    data["offsets"] = data["offsets"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])    
    
    return None, data

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
        few_shot_prompt = get_few_shot_prompt(
            examples=FEW_SHOT_EXAMPLES,
            seed=seed
        )
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
SMDS = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}
