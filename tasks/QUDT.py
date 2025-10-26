###############################################################################
# Imports
import os
import re
import pandas as pd
import random
import json

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.utils import FEW_SHOT_PROMPT_TEMPLATE, calc_metrics_extraction, process_responses_extraction

###############################################################################

###############################################################################
# Constants
LABEL2ID = {
    "O": 0,
    "QUOTE": 1,
    "TITLE": 2
    # ,"OP": 3,
    # "CP": 4,
    # "GEN_SRC": 5,
}
LABELS = list(LABEL2ID.keys())

DATA_DIR = os.path.join(r'P:\downstream_tasks\tib_citations\converted')

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

# PROMPTS = {
#     # "system": """You are a professional linguist specializing in ancient Tibetan text corpora.\nYour current task is citation detection: detect if an Explicit Citation exists, and if so, label it.\nRecall that explicit citations are sometimes accompanied by specific particles or phrases that indicate the presence of a citation.\nFor example - las, nas, la don (gsungs pa), ji skad du, na re, las shes te, la 'byung ba, las kyang, de ltar yang, ergative (after mentioning an author), pas bka' stsal pa.\nOr closing particles such as ces/shes/zhes (bya ba / verb such as gsungs + pa/ba + particle, in case of gsungs pa lta bu’o/bzhin no, do include the lta bu’o, and if there is a longer closing phrase, for example “zhes ji skad du gsungs pa bzhin no/” then mark the whole phrase), zhes gsungs so, zhe'o, zhe na (only when it is clearly a citation).\nA sample might have 0, 1 or even multiple citations.\nYour output should be a list of tuples, of the form [("QUOTE", "text span0"),("QUOTE", "text span1"),("QUOTE", "text span2"),("QUOTE", "text span3")] for each detected citation in a sample.\nThe text span should contain the FULL quote text, and you should not include ANYTHING ELSE except for the labels.\nDo NOT, I repeat, do NOT include anything outside of the defined format. Only label something if you are 100% certain - it's possible and likely there are no quotes.""",
#     # "system": """You are a professional Tibetan linguist specializing in ancient Tibetan. You are currently tasked with Explicit Citation dectection. An "Explicit Citation" is a quote, which explicitly states the source of the quote, such as an author, title, or other identifying information. Your task is to detect if an Explicit Citation exists in the given Tibetan text sample, and if so, label it.\nRecall that explicit citations are sometimes accompanied by specific particles or phrases that indicate the presence of a citation.\nFor example - las, nas, la don (gsungs pa), ji skad du, na re, las shes te, la 'byung ba, las kyang, de ltar yang, ergative (after mentioning an author), pas bka' stsal pa.\nOr closing particles such as ces/shes/zhes (bya ba / verb such as gsungs + pa/ba + particle, in case of gsungs pa lta bu’o/bzhin no, do include the lta bu’o, and if there is a longer closing phrase, for example “zhes ji skad du gsungs pa bzhin no/” then mark the whole phrase), zhes gsungs so, zhe'o, zhe na (only when it is clearly a citation).\nA sample might have 0, 1 or even multiple citations.\nYour output should be a list of tuples, of the form [("QUOTE", "text span0"),("QUOTE", "text span1"),("QUOTE", "text span2"),("QUOTE", "text span3")] for each detected citation in a sample.\nThe text span should contain the FULL quote text, and you should not include ANYTHING ELSE except for the labels.\nDo NOT, I repeat, do NOT include anything outside of the defined format. Only label something if you are 100% certain - it's possible and likely there are no quotes.""",
#     # "system": """The task is to label explicit citations in the given Tibetan sentence. Below are some examples with Input and Output pairs. For the prediction, you should generate the output in the same format as in the examples. Do not give any explanations. A sample may have 0, 1 or even multiple citations. \n\nSample:"/yang de nyid las gsungs pa/ gang chos kyi tshogs la sbyor ba de ni de'i ye shes kyi tshogs yin no/"\nOutput:[("QUOTE","gang chos kyi tshogs la sbyor ba de ni de'i ye shes kyi tshogs yin no/")]\n\nSample:"/sems can dag pa de ni ma zhum song/"\nOutput:[]\n\nSample:"/gang gi sgrib pa med pa la ltung ba gnas pa de ni gnas ma yin no zhes gsungs so/"\nOutput:[("QUOTE", "/gang gi sgrib pa med pa la ltung ba gnas pa de ni gnas ma yin no")]\n\nSample:"/blo mchog de dag mchog tu gzugs bzang ba/ /gzugs kyi dam pa yongs su ston byed cing/ /gzugs la mos pa'i sems can 'dul bar byed/ /de dag ka la ping ka'i sgra snyan dang/ /khu byug kun la ngang pa'i sgra skad dang/"\nOutput:[]\n\n""",
#     "system": """The task is to label explicit citations in the given Tibetan sentence. Below are some examples with Input and Output pairs. For the prediction, you should generate the output in the same format as in the examples. Do not give any explanations. A sample may have 0, 1 or even multiple citations. Recall that explicit citations are often preceded by opening phrases such as (but not limited to) las, nas, la don (followed by gsungs pa), ji skad du, na re, las shes te, etc.; they are often followed by closing phrases such as ces, shes, zhes, etc.\n\nSample:"/yang de nyid las gsungs pa/ gang chos kyi tshogs la sbyor ba de ni de'i ye shes kyi tshogs yin no/"\nOutput:[("QUOTE","gang chos kyi tshogs la sbyor ba de ni de'i ye shes kyi tshogs yin no/")]\n\nSample:"/sems can dag pa de ni ma zhum song/"\nOutput:[]\n\nSample:"/gang gi sgrib pa med pa la ltung ba gnas pa de ni gnas ma yin no zhes gsungs so/"\nOutput:[("QUOTE", "/gang gi sgrib pa med pa la ltung ba gnas pa de ni gnas ma yin no")]\n\nSample:"/blo mchog de dag mchog tu gzugs bzang ba/ /gzugs kyi dam pa yongs su ston byed cing/ /gzugs la mos pa'i sems can 'dul bar byed/ /de dag ka la ping ka'i sgra snyan dang/ /khu byug kun la ngang pa'i sgra skad dang/"\nOutput:[]\n\n""",
#     "user": "Sample:{sample}\n",
# }
PROMPTS = {
    "system": """You are a professional linguist and philologist specializing in ancient Tibetan. You are tasked with finding citations written in Tibetan: 
Your task is to analyze a given text written in Tibetan and extract any QUOTE it contains.

Definitions:
A QUOTE: a short sentence or paragraph citation referencing another work written in Tibetan.
Recall that QUOTEs are often preceded by phrases such as: ལས་, ནས་, ལ་དོན་(གསུངས་པ་), ཇི་སྐད་དུ་, ན་རེ་, ལས་ཤེས་ཏེ་, ལ་འབྱུང་བ་, ལས་ཀྱང་, དེ་ལྟར་ཡང་, etc.
Recall that QUOTEs are often followed by phrases such as: ཅེས་/ཤེས་/ཞེས་བྱ་བ(འོ་), ཅེས་/ཤེས་/ཞེས་གསུངས་པ(འོ་), ཅེས་/ཤེས་/ཞེས་གསུངས་སོ་, ཅེས་/ཤེས་/ཞེས་གསུངས་པ་ལྟ་བུའོ་, ཅེས་/ཤེས་/ཞེས་གསུངས་པ་བཞིན་ནོ་, ཅེས་/ཤེས་/ཞེས་ཇི་སྐད་དུ་གསུངས་པ་བཞིན་ནོ་, ཞེའོ་, ཅེ་/ཞེ་ན་ (only when it is clearly a citation and not just an objection or something similar), etc."
If there is a longer closing phrase, for example “zhes ji skad du gsungs pa bzhin no/” then mark the whole phrase.

Annotation Guideline:
- A sample might have 0, 1 or even multiple citations.
- Think carefully about cultural and topical relevance in ancient Tibetan. 
- If detected, write the QUOTE exactly as it appears in the sentence, without any changes. Only answer in JSON.

Return your output as a JSON with "prediction" key. The value is a list of dictionaries, each with:
- LABEL: "QUOTE"
- SPAN: the exact text span that contains the simile (minimal required span)

Example for an item: {{"LABEL": "QUOTE", "SPAN": "ཞེས་གསུངས་སོ"}}

If no citations is found, return an empty list under the "prediction" key.
Only respond with the JSON output, do not include any additional text or explanations.
""",
    "user": "Text: {text}\n"
}


# Return your output as a JSON with "prediction" key. The value is a list of lists [LABEL, SPAN], each with:
# - LABEL: should always be "QUOTE".
# - SPAN: the exact text span that contains the citation (minimal required span), exactly as it appears in the input text.
#
# If no citations are found, return an empty list under the "prediction" key.
# Only respond with the JSON output with list of size 0. Do NOT include any additional text or explanations.
# """,
#     "user": "Text: {text}\n"
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

    test_path = os.path.join(data_dir, "Tibetan/QUDT/test.jsonl")
    train_path = os.path.join(data_dir, "Tibetan/QUDT/train.jsonl")
    
    if os.path.exists(test_path):
        test = pd.read_json(test_path, lines=True)
        if os.path.exists(train_path):
            train = pd.read_json(train_path, lines=True)
        else:
            print("train file does not exist")

    return train, test

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

def process_tibetan_citation_responses_extraction(
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
        has_hallucinated = False
        try:
            if resp and "responses" in resp:
                responses = resp["responses"]
                for response in responses:
                    if isinstance(response, str):
                        if response.startswith('```json'):
                            # Strip Markdown markers and parse JSON
                            json_str = response.strip('`').replace('json\n', '', 1).strip()
                            response = json.loads(json_str)
                    if "prediction" in response:
                        response = response["prediction"]

                        if isinstance(response, list):
                            for citation in response:
                                add_citation_predictions(response_predictions, citation)
                        elif isinstance(response, dict):
                            add_citation_predictions(response_predictions, citation)
                        else:
                            print(f"Unsupported response type:[{type(response)}] for response:[{response}]")
                            has_hallucinated = True
                    else:
                        print(f"No 'prediction' key in response: {response}")
                        has_hallucinated = True


            # if len(response_predictions) == 0:
            #     hallucinated.append(True)
            # else:
            hallucinated.append(has_hallucinated)

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
    csv_results, metrics = calc_metrics_extraction(ground_truth, predictions, texts, verbose=False)
    metrics["hallucinations"] = int(sum(hallucinated))
    csv_results["hallucinations"] = metrics["hallucinations"]

    # Add predictions and hallucination flags to DataFrame
    test[pred_col] = predictions
    test["hallucinated"] = hallucinated

    return csv_results, metrics, test, None

def add_citation_predictions(preds, citation):
    add_citation_prediction(preds, citation, "QUOTE")

def add_citation_prediction(preds, citation, citation_type):
    if citation_type in citation:
        label = citation[0]
        span = citation[1]
        if label is not None:
            if label != "" and label != "Unknown":
                pred = (citation_type, span)
                preds.append(pred)

process_responses = process_responses_extraction # process_tibetan_citation_responses_extraction

###############################################################################

###############################################################################
# Export
QUDT = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}
