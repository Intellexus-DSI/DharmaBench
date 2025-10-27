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
LABEL2ID = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'L6': 6, 'L7': 7, 'L8': 8, 'L9': 9, "EMPTY": -1}

LABELS = list(LABEL2ID.keys())

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Columns to be logged together with the model's raw responses
LOG_COLUMNS = ["text", "label"]

###############################################################################


###############################################################################
# Prompts
FEW_SHOT_EXAMPLES = [
        {
            "input": "paramārthaśāstrakṛtyā kurvāṇaṃ śāstṛkṛtyam iva loke | yaṃ buddhimatām agryaṃ dvitīyam iva buddham ity āhuḥ ||",
            "output": {
                "label": "āryā"
            }
        },
        {
            "input": "paramārthaśāstrakṛtyā kurvāṇaṃ śāstṛkṛtyam iva loke | yaṃ buddhimatām agryaṃ dvitīyam iva buddham ity āhuḥ ||",
            "output": {
                "label": "āryā"
            }
        },
        {
            "input": "tena vasubaṃdhunāmnā bhaviṣyaparamārthabandhunā jagataḥ | abhidharmapratyāsaḥ kṛto 'yam abhidharmakośākhyaḥ ||",
            "output": {
                "label": "āryā"
            }
        },
        {
            "input": "tena vasubaṃdhunāmnā bhaviṣyaparamārthabandhunā jagataḥ | abhidharmapratyāsaḥ kṛto 'yam abhidharmakośākhyaḥ ||",
            "output": {
                "label": "āryā"
            }
        },
    ]
# change to your task prompts


# PROMPTS = {
#     "system": """You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse.
#     This is a definition of Sanskrit meter: 'The rhythmic structure of Sanskrit verse, determined by the pattern of guru (heavy/long) and laghu (light/short) syllables, following traditional chandas patterns'.
#     Analyze the syllable quantities (guru-laghu patterns) and classify the verse into one of these specific chandas categories:
#     0. "anuṣṭubh_pathyā": 32-syllable meter, standard anuṣṭubh with regular patterns in all quarters.
#     1. "anuṣṭubh_vipulā": 32-syllable meter, irregular anuṣṭubh with unusual syllable patterns in odd quarters.
#     2. "āryā"
#     3. "drutavilambita"
#     4. "mandākrāntā"
#     5. "sragdharā"
#     6. "śālinī"
#     7. "śārdūlavikrīḍita"
#     8. "upajāti_family": 44-syllable meter, verses with 11 syllables per quarter (not 8).
#     9. "vasantatilakā": 56-syllable meter, verses with 14 syllables per quarter (longest common meter).
#     10. "EMPTY"

#     IMPORTANT: You must use ONLY these exact 11 labels above. Do not create any new labels or variations.
#     Consider the pada (quarter-verse) structure and analyze the guru-laghu patterns carefully.
#     You are given one Sanskrit verse, you are an expert in Sanskrit prosody and chandas classification.

#     CRITICAL RESPONSE FORMAT:
#     - Do NOT provide any reasoning, explanation, or analysis
#     - Do NOT include any text before or after the JSON
#     - Return ONLY this exact format: {{"label": "EXACT_LABEL_NAME"}}
#     - Replace EXACT_LABEL_NAME with one of the 11 labels above
#     - No additional fields, no explanations, no other content

#     Example response: {{"label": "anuṣṭubh_vipulā"}}
#     """,
#     "user": "Verse:{verse}\n",
# }


# PROMPTS = {
#     "system": """You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse.
#     This is a definition of Sanskrit meter: 'The rhythmic structure of Sanskrit verse, determined by the pattern of guru (heavy/long) and laghu (light/short) syllables, following traditional chandas patterns'.
#     Analyze the syllable quantities (guru-laghu patterns) and classify the verse into one of these specific chandas categories:
#     0. "anuṣṭubh_pathyā": A 32-syllable Sanskrit meter with four 8-syllable quarters. General constraints: syllables 1 and 8 of each quarter can be long or short; syllables 2-3 cannot both be short; in quarters 2 and 4, syllables 2-4 must not follow long-short-long pattern and syllables 5-7 must be short-long-short. The defining characteristic is that syllables 5-7 in quarters 1 and 3 must follow the pattern short-long-long.
#     1. "anuṣṭubh_vipulā": A 32-syllable Sanskrit meter with four 8-syllable quarters following the same general constraints as pathyā. The distinguishing feature is that in at least one of quarters 1 or 3, syllables 5-7 must follow one of four specific patterns: short-short-short, long-short-short, long-long-long (requiring caesura after syllable 5), or long-short-long (requiring caesura after syllable 4).
#     2. "āryā"
#     3. "drutavilambita"
#     4. "mandākrāntā"
#     5. "sragdharā"
#     6. "śālinī"
#     7. "śārdūlavikrīḍita"
#     8. "upajāti_family":  This parent-class of metres consist of 44 syllables divided in to 4 even quarters. The first and the last syllables can be either long or short. And the 2nd-10th must follow the pattern: long-short-long-long-short-short-long-short-long.
#     9. "vasantatilakā": Vasantatilaka meter (14 syllables per quarter = 56 total) - CRITICAL: Pattern T-B-J-J-G-G, exactly 14 syllables per quarter
#     10. "EMPTY"

#     IMPORTANT: You must use ONLY these exact 11 labels above. Do not create any new labels or variations.
#     Consider the pada (quarter-verse) structure and analyze the guru-laghu patterns carefully.
#     You are given one Sanskrit verse, you are an expert in Sanskrit prosody and chandas classification.

#     CRITICAL RESPONSE FORMAT:
#     - Do NOT provide any reasoning, explanation, or analysis
#     - Do NOT include any text before or after the JSON
#     - Return ONLY this exact format: {{"label": "EXACT_LABEL_NAME"}}
#     - Replace EXACT_LABEL_NAME with one of the 11 labels above
#     - No additional fields, no explanations, no other content

#     Example response: {{"label": "anuṣṭubh_vipulā"}}
#     """,
#     "user": "Verse:{verse}\n",
# }

# PROMPTS = {
#     "system": """You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse.
#     This is a definition of Sanskrit meter: 'The rhythmic structure of Sanskrit verse, determined by the pattern of guru (heavy/long) and laghu (light/short) syllables, following traditional chandas patterns'.
#     Analyze the syllable quantities (guru-laghu patterns) and classify the verse into one of these specific chandas categories:
#     0. "anuṣṭubh_pathyā"
#     1. "anuṣṭubh_vipulā": Anushtubh meter with vipula variation (8+8+8+8 syllables) - CRITICAL: Contains guru (heavy) syllable in position 6 or 7 of odd quarters
#     2. "āryā"
#     3. "drutavilambita"
#     4. "mandākrāntā"
#     5. "sragdharā"
#     6. "śālinī"
#     7. "śārdūlavikrīḍita"
#     8. "upajāti_family": Upajati meter family (11 syllables per quarter = 44 total) - Mix of Indravajra/Upendravajra
#     9. "vasantatilakā": Vasantatilaka meter (14 syllables per quarter = 56 total) - CRITICAL: Pattern T-B-J-J-G-G, exactly 14 syllables per quarter
#     10. "EMPTY"

#     IMPORTANT: You must use ONLY these exact 11 labels above. Do not create any new labels or variations.
#     Consider the pada (quarter-verse) structure and analyze the guru-laghu patterns carefully.
#     You are given one Sanskrit verse, you are an expert in Sanskrit prosody and chandas classification.
#     Classify the verse using only the labels above and provide confidence level with brief explanation.

#     Return the meter as 'label' that fits the most to the input verse according to the meter name.
#     Only answer with the label, in JSON format with one field "label" and the value being the EXACT label name from the list of 11 labels above.
#     """,
#     "user": "Verse:{verse}\n",
# }

# PROMPTS = {
#     "system": """You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse.
#     This is a definition of Sanskrit meter: 'The rhythmic structure of Sanskrit verse, determined by the pattern of guru (heavy/long) and laghu (light/short) syllables, following traditional chandas patterns'.
#     Analyze the syllable quantities (guru-laghu patterns) and classify the verse into one of these specific chandas categories:
#     0. "anuṣṭubh_pathyā": Standard anushtubh meter with pathya form
#     1. "anuṣṭubh_vipulā": Anushtubh meter with vipula variation
#     2. "āryā": Arya meter
#     3. "drutavilambita": Drutavilambita meter
#     4. "mandākrāntā": Mandakranta meter
#     5. "sragdharā": Sragdhara meter
#     6. "śālinī": Shalini meter
#     7. "śārdūlavikrīḍita": Shardulavikridita meter
#     8. "upajāti_family": Upajati meter family
#     9. "vasantatilakā": Vasantatilaka meter
#     10. "EMPTY": When the meter cannot be determined

#     IMPORTANT: You must use ONLY these exact 11 labels above. Do not create any new labels or variations.
#     Consider the pada (quarter-verse) structure and analyze the guru-laghu patterns carefully.
#     You are given one Sanskrit verse, you are an expert in Sanskrit prosody and chandas classification.
#     Classify the verse using only the labels above and provide confidence level with brief explanation.

#     Return the meter as 'label' that fits the most to the input verse according to the meter name.
#     Only answer with the label, in JSON format with one field "label" and the value being the EXACT label name from the list of 11 labels above.
#     """,
#     "user": "Verse:{verse}\n",
# }

# PROMPTS = {
#     "system": """You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse.
#     This is a definition of Sanskrit meter: 'The rhythmic structure of Sanskrit verse, determined by the pattern of guru (heavy/long) and laghu (light/short) syllables, following traditional chandas patterns'.
#     Analyze the syllable quantities (guru-laghu patterns) and classify the verse into one of these specific chandas categories:
#     0. "anuṣṭubh_pathyā": 32-syllable meter, standard anuṣṭubh with regular patterns in all quarters.
#     1. "anuṣṭubh_vipulā": 32-syllable meter, irregular anuṣṭubh with unusual syllable patterns in odd quarters.
#     2. "āryā": Arya meter (12+18+12+15 mātrās) - Quantitative meter, characteristic cadence
#     3. "drutavilambita": Drutavilambita meter (12 syllables per quarter = 48 total)
#     4. "mandākrāntā": Mandakranta meter (17 syllables per quarter = 68 total)
#     5. "sragdharā": Sragdhara meter (21 syllables per quarter = 84 total)
#     6. "śālinī": Shalini meter (11 syllables per quarter = 44 total)
#     7. "śārdūlavikrīḍita": Shardulavikridita meter (19 syllables per quarter = 76 total)
#     8. "upajāti_family": 44-syllable meter, verses with 11 syllables per quarter (not 8).
#     9. "vasantatilakā": 56-syllable meter, verses with 14 syllables per quarter (longest common meter).
#     10. "EMPTY": When the meter cannot be determined

#     IMPORTANT: You must use ONLY these exact 11 labels above. Do not create any new labels or variations.
#     Consider the pada (quarter-verse) structure and analyze the guru-laghu patterns carefully.
#     You are given one Sanskrit verse, you are an expert in Sanskrit prosody and chandas classification.

#     CRITICAL RESPONSE FORMAT:
#     - Do NOT provide any reasoning, explanation, or analysis
#     - Do NOT include any text before or after the JSON
#     - Return ONLY this exact format: {{"label": "EXACT_LABEL_NAME"}}
#     - Replace EXACT_LABEL_NAME with one of the 11 labels above
#     - No additional fields, no explanations, no other content

#     Example response: {{"label": "anuṣṭubh_vipulā"}}
#     """,
#     "user": "Verse:{verse}\n",
# }



# """You are a computational linguist, philologist, and scholar of Sanskrit textual traditions. 
#     Your task is to analyze a given root-text and a given commentary written in Sanskrit and decide if the commentary is indeed a commentary of the root.
#     Classification criteria:
#     - TRUE: The commentary is relevant and provides insight into the root-text.
#     - FALSE: The commentary is not relevant or does not provide insight into the root-text.

#     Response Guideline:
#     Return your output as a JSON with a "label" key. The value is either "TRUE" or "FALSE". For example
#     - "label": "TRUE"
#     or
#     - "label": "FALSE"
#     Only respond with the JSON output, do not include additional text or explanations.
#     """,


PROMPTS = {
"system": """
You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse written in Sanskrit.
This is a definition of Sanskrit meter: 'The rhythmic structure of Sanskrit verse, determined by the pattern of guru (heavy/long) and laghu (light/short) syllables, following traditional chandas patterns'.
Analyze the syllable quantities (guru-laghu patterns) and classify the verse into one of these specific chandas labels:
"L0": "anuṣṭubh_pathyā", 32-syllable meter, standard anuṣṭubh with regular patterns in all quarters.
"L1": "anuṣṭubh_vipulā", 32-syllable meter, irregular anuṣṭubh with unusual syllable patterns in odd quarters.
"L2": "āryā", Arya meter (12+18+12+15 mātrās) - Quantitative meter, characteristic cadence
"L3": "drutavilambita", Drutavilambita meter (12 syllables per quarter = 48 total)
"L4": "mandākrāntā", Mandakranta meter (17 syllables per quarter = 68 total)
"L5": "sragdharā", Sragdhara meter (21 syllables per quarter = 84 total)
"L6": "śālinī", Shalini meter (11 syllables per quarter = 44 total)
"L7": "śārdūlavikrīḍita", Shardulavikridita meter (19 syllables per quarter = 76 total)
"L8": "upajāti_family", 44-syllable meter, verses with 11 syllables per quarter (not 8).
"L9": "vasantatilakā", 56-syllable meter, verses with 14 syllables per quarter (longest common meter).

IMPORTANT: You must use ONLY the exact index numbers above. Do not use label names or create variations.
Consider the pada (quarter-verse) structure and analyze the guru-laghu patterns carefully.
You are given one Sanskrit verse, you are an expert in Sanskrit prosody and chandas classification.

CRITICAL RESPONSE FORMAT:
- Do NOT provide any reasoning, explanation, or analysis
- Do NOT include any text before or after the JSON
- Return ONLY this exact format: {{"label": "INDEX_NUMBER"}}
- Replace INDEX_NUMBER with the corresponding index (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9)
- No additional fields, no explanations, no other content
- Do not make any calculations, the numbers are just indexes! 

Example response: {{"label": "L1"}}
Only respond with the JSON output, do not include additional text or explanations.
""",
"user": "Verse:{verse}\n",
}



# PROMPTS = {
#     "system": """You are a professional prosodist specializing in Sanskrit chandas (metrical analysis) and your task is to classify the metrical pattern of a given Sanskrit verse.

#     STEP 1: Count the total syllables in the verse
#     STEP 2: Divide by 4 to get syllables per quarter
#     STEP 3: Match to the correct meter based on syllable count

#     Classify the verse into one of these chandas categories:
#     0. "anuṣṭubh_pathyā": EXACTLY 8 syllables per quarter (32 total) - Regular pattern
#     1. "anuṣṭubh_vipulā": EXACTLY 8 syllables per quarter (32 total) - Irregular pattern
#     2. "āryā": Variable syllable count with mātrā-based rhythm
#     3. "drutavilambita": EXACTLY 12 syllables per quarter (48 total)
#     4. "mandākrāntā": EXACTLY 17 syllables per quarter (68 total)
#     5. "sragdharā": EXACTLY 21 syllables per quarter (84 total)
#     6. "śālinī": EXACTLY 11 syllables per quarter (44 total)
#     7. "śārdūlavikrīḍita": EXACTLY 19 syllables per quarter (76 total)
#     8. "upajāti_family": EXACTLY 11 syllables per quarter (44 total)
#     9. "vasantatilakā": EXACTLY 14 syllables per quarter (56 total)
#     10. "EMPTY": When meter cannot be determined

#     CRITICAL: If syllables per quarter = 8, choose between 0 or 1. If = 11, choose between 6 or 8. If = 14, choose 9.

#     IMPORTANT: You must use ONLY the exact index numbers above. Do not use label names or create variations.

#     CRITICAL RESPONSE FORMAT - FOLLOW EXACTLY:
#     - Do NOT provide any reasoning, explanation, analysis, or additional text
#     - Do NOT include any text before or after the JSON
#     - Do NOT explain your thinking process
#     - Do NOT add any commentary
#     - Return ONLY this exact format: {{"label": "INDEX_NUMBER"}}
#     - Replace INDEX_NUMBER with the corresponding index (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10)
#     - No additional fields, no explanations, no other content whatsoever

#     Example response: {{"label": "1"}}

#     ABSOLUTELY NO TEXT OTHER THAN THE JSON RESPONSE.
#     """,
#     "user": "Verse:{verse}\n",
# }



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

    path = os.path.join(data_dir, "Sanskrit/MCS/test.jsonl")
    if os.path.exists(path):
        df = pd.read_json(path, lines=True)
        if 'label' in df.columns:
            df['label'] = "L" + df['label'].astype(str)
        elif 'meter_normalized' in df.columns:
            df['label'] = "L" + df['meter_normalized'].astype(str)
        elif 'meter' in df.columns:
            df['label'] = "L" + df['meter'].astype(str)

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

    ########################################
    # TODO: Add logic to create examples of few-shot examples
    # Get data
    # train, _ = get_data()
    # train, _ = get_data()
    # Get shot examples for the given language
    # few_shot_examples = train.sample(n=shots, random_state=seed)
    # few_shot_examples = train.sample(n=shots, random_state=seed)

    # examples = [
    #     {
    #         "input": f"text: {row['text']}",
    #         "output": {"label": row["label"]},
    #     }
    #     for _, row in few_shot_examples.iterrows()
    # ]
    # examples = [
    #     {
    #         "input": f"text: {row['text']}",
    #         "output": {"label": row["label"]},
    #     }
    #     for _, row in few_shot_examples.iterrows()
    # ]

    # ########################################

    # # Shuffle the examples
    # random.shuffle(examples)  # Shuffle the list in place
    # # Shuffle the examples
    # random.shuffle(examples)  # Shuffle the list in place

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=FEW_SHOT_EXAMPLES,
    )
    return few_shot_prompt

def get_prompt(config: dict, **kwargs) -> ChatPromptTemplate:
    """
    Get the prompt for the given task and prompt type.
    Also returns the schema for the task if the model supports structured output.
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
        {"verse": row["text"]}
        for _, row in data.iterrows()
    ]
    return user_inputs

process_responses = process_responses_classification

###############################################################################

###############################################################################
# Export
MCS = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}