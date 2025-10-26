###############################################################################
# Imports
import os
import re
import pandas as pd
import random
import json
import numpy as np

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.utils import FEW_SHOT_PROMPT_TEMPLATE, process_responses_classification, process_responses_extraction, self_consistency_classification, calc_metrics_classification

###############################################################################

###############################################################################
# Constants
LABEL2ID = {
    "EMPTY": -1,  # Special label for empty responses
    "Verse": 0,
    "Prose": 1,
}
LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(parent_dir, "data/tibetan/verses_vs_prose")

# Columns to be logged together with the model's raw responses
# Note: DharmaBench uses 'id' instead of 'sample_id'
LOG_COLUMNS = ["id", "text", "label"]

###############################################################################


###############################################################################
# Prompts

# FEW_SHOT_EXAMPLES = {s
#     "english": {
#         "cot": [
#             {
#                 "input": "sentence: The very solicitors' boys who have kept the wretched suitors at bay, by protesting time out of mind that Mr Chizzle, Mizzle, or otherwise was particularly engaged and had appointments until dinner, may have got an extra moral twist and shuffle into themselves out of Jarndyce and Jarndyce.",
#                 "output": """sentence: The very solicitors' boys who have kept the wretched suitors at bay, by protesting time out of mind that Mr Chizzle, Mizzle, or otherwise was particularly engaged and had appointments until dinner, may have got an extra moral twist and shuffle into themselves out of Jarndyce and Jarndyce.
# explanation: Time out of mind" is idiomatic, meaning for an extremely long time rather than a specific duration. Here, it emphasizes the endless nature of the legal case.
# idioms: ['time out of mind']""",
#             },
#             {
#                 "input": "sentence: In 1746, some months after his 36th birthday, Samuel Johnson, that great literary figure of the 18th century, affectionately referred to as the Good Doctor, began work on his monumental Dictionary of the English Language.",
#                 "output": """sentence: In 1746, some months after his 36th birthday, Samuel Johnson, that great literary figure of the 18th century, affectionately referred to as the Good Doctor, began work on his monumental Dictionary of the English Language.
# explanation: The Good Doctor" is idiomatic because it serves as an affectionate, honorific nickname rather than a literal description of Johnson’s medical expertise. It conveys respect and familiarity, commonly used for well-regarded scholars or physicians.
# idioms: ['the Good Doctor']""",
#             },
#             {
#                 "input": "sentence: The film briefly visits a “rubber room” in New York City where idle teachers accused of misconduct wait months and sometimes years for hearings while drawing full salaries.",
#                 "output": """sentence: The film briefly visits a “rubber room” in New York City where idle teachers accused of misconduct wait months and sometimes years for hearings while drawing full salaries.
# explanation: Rubber room" is idiomatic because it does not refer to a room made of rubber but rather a slang term for a detention-like space where teachers accused of misconduct are sent while awaiting hearings. The phrase carries a figurative meaning, implying a bureaucratic limbo or an absurd, almost surreal situation.
# idioms: ['rubber room']""",
#             },
#             {
#                 "input": "sentence: We have no decisions in our state directly on point. With us the problem is one of first impression. None of the cases cited is on point.",
#                 "output": """sentence: We have no decisions in our state directly on point. With us the problem is one of first impression. None of the cases cited is on point.
# explanation: There are no idioms in this sentence because "on point" and "one of first impression" are legal terms used in their precise, technical sense rather than figuratively. They retain their literal legal meanings.
# idioms: []""",
#             },
#             {
#                 "input": "sentence: To find the real Michel Foucault is to ask “which one”?, Should we look at the life of the man himself, who as a boy wanted to be a goldfish, but became a philosopher and historian, political activist, leather queen, bestseller, tireless campaigner for dissident causes?",
#                 "output": """sentence: To find the real Michel Foucault is to ask “which one”?, Should we look at the life of the man himself, who as a boy wanted to be a goldfish, but became a philosopher and historian, political activist, leather queen, bestseller, tireless campaigner for dissident causes?
# explanation: There are no idioms in this sentence because all phrases, including "wanted to be a goldfish" and "leather queen," are used either literally, descriptively, or as personal characterizations rather than established idiomatic expressions with non-literal meanings.
# idioms: []""",
#             },
#             {
#                 "input": "sentence: Compared to other modern languages, C is much less forgiving, much more terse, and generally much more ill tempered. However, it is about as close to programming on the bare metal as you can get while still using a well-supported language with good library support for everything from numeric calculations to graphics.",
#                 "output": """sentence: Compared to other modern languages, C is much less forgiving, much more terse, and generally much more ill tempered. However, it is about as close to programming on the bare metal as you can get while still using a well-supported language with good library support for everything from numeric calculations to graphics.
# explanation: There are no idioms in this sentence because "on the bare metal" is a common technical phrase in computing, used in its literal sense within the field. It describes programming close to hardware rather than serving a metaphorical or figurative purpose.
# idioms: []""",
#             },
#         ],
#     }
# }

PROMPTS = {
    "system":
        """
        You are a computational linguist, philologist, and scholar of Classical Tibetan textual traditions. 
        
        Your task is to analyze a given passage or verse written in Tibetan and classify it as either Verse or Prose based on its predominant textual structure.
        
        Classification criteria:
        - Verse: Text with regular meter, line breaks, or poetic structure (including traditional Tibetan verse forms)
        - Prose: Continuous narrative or expository text without regular metrical patterns
        
        Annotation Guideline:
        If the passage contains both forms, classify based on whichever comprises more than 50% of the content.
        
        Response Guideline:
        Return your output as a JSON with a "label" key. The value is either "Verse" or "Prose". For example
        - "label": "Verse"
        or 
        - "label": "Prose"
        Only respond with the JSON output, do not include additional text or explanations.
        """,
    "user": "Text: {text}\n"
}

###############################################################################



###############################################################################

def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = os.path.join(data_dir, "Tibetan/VPCT/test.jsonl")
    train = os.path.join(data_dir, "Tibetan/VPCT/train.jsonl")
    
    if os.path.exists(test):
        test = pd.read_json(test, lines=True)
        if os.path.exists(train):
            train = pd.read_json(train, lines=True)
        else:
            print("train file does not exist")
        
        if 'class' in test.columns:
            test['label'] = test['class']
        if not train.empty and 'class' in train.columns:
            train['label'] = train['class']
        
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

    ########################################
    # Get data
    random.seed(seed)
    train, _ = get_data()
    sample_df = train.sample(min(shots, len(train)), random_state=seed)
    examples = [
        {
            "input":  f"Text: {row['text']}",
            "output": {"label": row["label"]}
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

    #print('shay tib_verses_vs_prose prompt start')
    #print(prompt)
    #print('shay tib_verses_vs_prose prompt end')

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
VPCT = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}
