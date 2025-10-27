"""
Utilities functions.
"""

####################################################################################################
# Imports
import os
import logging
from typing import Union
import json
import re
import pandas as pd
from collections import Counter
import smtplib
from email.mime.text import MIMEText
import yaml
import eval4ner.muc as muc
from ast import literal_eval
import random

from pandas import DataFrame
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score, classification_report,
)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from sympy.physics.control import Series
from typing import TypedDict
from pydantic import BaseModel
from langchain_core.messages import AIMessage


####################################################################################################


####################################################################################################
# Constants

FEW_SHOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

MERGE_COLUMNS = ["debug", "model", "prompt_type", "seed", "sc_runs", "temperature", "shots"]

####################################################################################################


####################################################################################################
# Functions

def set_keys(keys: dict):
    """
    Set API keys as environment variables.
    :param keys: dictionary with keys
    """
    for key, value in keys.items():
        os.environ[key] = value


def get_logger(name):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding="utf-8",
    )
    logger = logging.getLogger(name)
    return logger


def read_tsv(file_path):
    data = pd.read_csv(file_path, sep="\t", quoting=3)
    return data


def run_batch(chain, user_inputs, run_index, logger):
    """Execute batch processing."""
    try:
        raw_responses = chain.batch(user_inputs)
        logger.info(f"Generated {len(raw_responses)} raw responses for run {run_index}")
        return raw_responses
    except Exception as e:
        logger.error(f"Error during batch run {run_index}: {e}")
        return [None] * len(user_inputs)


def run_individual(chain, user_inputs, run_index, exp_dir, logger):
    """Execute individual processing with error handling."""
    problematic_file = os.path.join(exp_dir, f"problematic_inputs_run_{run_index}.txt")

    with open(problematic_file, "w") as f:
        f.write("Problematic inputs:\n")

    def safe_invoke(i, row):
        try:
            print()
            print(f"invoking individual input {i} in run {run_index} with row [{row}]")
            print()
            invoke_result = chain.invoke(row)
            print(f'invoke_result : [{invoke_result}]')
            print()
            return invoke_result
        except Exception as e:
            logger.warning(f"Error for input {i} in run {run_index}: {e}")
            with open(problematic_file, "a") as f:
                f.write(f"Input {i}: {row}\nError: {e}\n\n")
            return None

    return [safe_invoke(i, row) for i, row in enumerate(user_inputs)]


def extract_ners_from_annotation(annotation_dict: dict) -> list:
    """
    Extracts named entities from the annotation dictionary.
    Args:
        annotation_dict (dict): The annotation dictionary containing results.
    Returns:
        list: A list of tuples containing (label, text) for each named entity.
    """
    ner_list = []
    if isinstance(annotation_dict, dict) and "result" in annotation_dict:
        for result in annotation_dict.get("result", []):
            value = result.get("value", {})
            labels = value.get("labels", [])
            text = value.get("text", "")

            for label in labels:  # usually only one, but safely looping
                ner_list.append((label, text))
    
    return ner_list

def _parse_json_manually(output: str) -> dict:
    """
    Parse reasoning output from the model, returning a dict with 'prediction'.
    Accepts non-strict formats (e.g., single quotes).
    """
    # Cut only after the reasoning part
    if "</think>" in output:
        output = output.split("</think>", 1)[-1]

    # remove mark-down code fencing
    clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.IGNORECASE | re.DOTALL)

    # 1️⃣ Try standard JSON
    try:
        data = json.loads(clean_json)
        if isinstance(data, dict) and "prediction" in data or "label" in data:
            return data
    except json.JSONDecodeError:
        pass

    # 2️⃣ Try Python-style literal (single quotes, etc.)
    try:
        data = literal_eval(clean_json)
        if isinstance(data, dict) and "prediction" in data or "label" in data:
            return data
    except (ValueError, SyntaxError):
        pass

    # 3️⃣ Last fallback: assume raw prediction content in plain text
    print("Failed to parse response completely, an empty dict will be returned.")
    return {}

def transform_extraction_prediction_to_tuples(data):
    try:
        return {
            'prediction': [
                [item.get('LABEL', ''), item.get('SPAN', '')]
                for item in data.get('prediction', [])
            ]
        }
    except (KeyError, TypeError):
        return {'prediction': []}

def fix_ext_preds(preds_dict):
    tuples = []
    for pred in preds_dict["prediction"]:
        # Convert all to dictionaries
        pred = dict(pred)
        # If prediction, parse non-serialized predictions
        # Convert all to dictionaries
        tuples.append((pred.get("LABEL", ""), pred.get("SPAN", "")))
    return {"prediction": tuples}



def parse_response(response) -> dict:
    if not response:
        print("")
        print("")
        print("Attention! received an empty response! Your model might be too busy and you need to run again!!!")
        print("")

    content = ""
    if "parsed" in response and response["parsed"]:
        print("Parsing 'parsed' field...")
        # If already a a dictionary, just return it
        if isinstance(response["parsed"], dict):
            if "prediction" in response["parsed"]:
                return fix_ext_preds(response["parsed"])
        else:
            try:
                ret = dict(response["parsed"])
                if "prediction" in ret:
                    return fix_ext_preds(ret)
                else:
                    return ret
                # # Check if 'label' contains a JSON string with 'prediction'
                # elif "label" in ret and isinstance(ret["label"], str):
                #     try:
                #         label_parsed = json.loads(ret["label"])
                #         if "prediction" in label_parsed:
                #             return fix_ext_preds(label_parsed)
                #         # If label is a list/dict but no 'prediction', wrap it
                #         elif isinstance(label_parsed, list):
                #             return fix_ext_preds({"prediction": label_parsed})
                #         else:
                #             return label_parsed
                #     except json.JSONDecodeError:
                        # return {"prediction": []}
                # return ret
            except Exception as e:
                print("Failed to process as scheme")

            # It is our schemas, convert to dict
            if isinstance(response["parsed"], str):
                try:
                    response["parsed"] = json.loads(response["parsed"])
                    if "prediction" in response["parsed"]:
                        return fix_ext_preds(response["parsed"])
                    else:
                            return response["parsed"]

                except Exception as e:
                    # Continue with the rest of the code
                    print(f"Error parsing 'parsed' field: {e}")
    else:
        if "raw" in response:
            content = response["raw"]
        else:
            content = response
        
        if hasattr(content, "content"):
            content = content.content
        
        # Try to loads
        try:
            ret = json.loads(content)
            if "prediction" in ret:
                ret = fix_ext_preds(ret)
            return ret
        except json.JSONDecodeError:
            print("Failed to decode JSON, trying to parse manually...")

        original_content = content  # Keep original for debugging

        # Strip code‑fences / prefixes
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
        content = re.sub(r"^Output:\s*", "", content, flags=re.IGNORECASE)

        # 1️⃣ Standard JSON
        try:
            parsed = json.loads(content)

            # Handling extraction response that looks like - {'prediction': [{'LABEL': 'Author', 'SPAN': 'paṇinaḥ'}]}
            if (isinstance(parsed, dict) and "prediction" in parsed) and (isinstance(parsed["prediction"], list)) and (len(parsed["prediction"]) > 0) and (("label" in parsed["prediction"][0]) or ("LABEL" in parsed["prediction"][0])):
                parsed = transform_extraction_prediction_to_tuples(parsed)

            if isinstance(parsed, dict) and "prediction" in parsed or "label" in parsed:
                ret = parsed
        except json.JSONDecodeError:
            pass

        # 2️⃣ NEW: Python‑style dict string (single quotes, etc.)
        try:
            maybe_dict = literal_eval(content)
            if isinstance(maybe_dict, dict) and ("prediction" in maybe_dict or "label" in maybe_dict):
                ret = maybe_dict
        except (ValueError, SyntaxError):
            pass

        # 3️⃣ Fallback parser
        try:
            print("Parsing response manually...")
            # print(f"Original content: {original_content}")
            ret = _parse_json_manually(content)
            print("Parse response manually!")

            if "prediction" in ret:
                return fix_ext_preds(ret)
            else:
                return ret
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {}


def self_consistency_classification(
    predictions: list[str],
    sc_runs: int = 1,
    min_occurrence_ratio: float = 0.5,
) -> Union[str, dict[str, float]]:
    """
    Calculate the most common prediction from multiple self-consistency runs.

    :param predictions: list of predictions from multiple SC runs
    :param sc_runs: number of self-consistency runs to consider
    :param min_occurrence_ratio: a minimum ratio of occurrences to consider a prediction valid
    :return: most common prediction or dictionary with predictions and their occurrence ratios
    """
    # Limit to specified number of SC runs
    predictions = predictions[:sc_runs]
    actual_runs = len(predictions)

    if actual_runs == 0:
        return {}

    # If only one run, return the prediction as is
    if actual_runs == 1:
        return {predictions[0]: 1.0} if predictions[0] is not None else {}

    # Count occurrences of each prediction
    prediction_counts = Counter(pred for pred in predictions if pred is not None)

    if not prediction_counts:
        return {}

    # Calculate occurrence ratios
    predictions_with_ratio = {
        prediction: count / actual_runs
        for prediction, count in prediction_counts.items()
    }

    # Filter by minimum occurrence ratio
    valid_predictions = {
        prediction: ratio
        for prediction, ratio in predictions_with_ratio.items()
        if ratio >= min_occurrence_ratio
    }
    # Sort by occurrence ratio in descending order
    valid_predictions = dict(
        sorted(valid_predictions.items(), key=lambda item: item[1], reverse=True)
    )

    return valid_predictions


def get_few_shot_prompt(
    examples: list[dict],
    seed: int = 42  
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot prompt form samples.
    :param examples: list of examples to use in the few-shot prompt
    :param seed: seed for reproducibility
    :return: few-shot prompt
    """
    random.seed(seed)  # Set the seed for reproducibility

    # Shuffle the examples
    random.shuffle(examples)  # Shuffle the list in place

    # Make sure samples are in json format, created from dictionary
    json_examples = []
    for example in examples:
        json_examples.append({
            "input": example["input"],
            "output": json.dumps(example["output"], ensure_ascii=False)
        })

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=json_examples,
    )
    return few_shot_prompt


def calc_metrics_classification(y_true: list, y_pred: list, label2id: dict, **kwargs) -> tuple[dict, dict, dict]:
    """
    Calculate metrics for classification
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :param label2id: dictionary mapping labels to ids
    :param kwargs: additional arguments (not used)
    :return: dictionary with metrics
    """
    print(f'y_true: {y_true[:5]}, y_pred: {y_pred[:5]}, label2id: {label2id}')

    # If the source of y_true is from a csv file then it's type at this point will be a pd.Series and it must be converted to a list.
    if isinstance(y_true, pd.Series):
        y_true = y_true.tolist()

    # If anything is string, convert to ids
    if isinstance(y_true[0], str):
        y_true = [label2id[label] for label in y_true if label in label2id]
    if isinstance(y_pred[0], str): # Use -1 for missing labels
        y_pred = [label2id[label] for label in y_pred if label in label2id]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="micro"),
        "recall": recall_score(y_true, y_pred, average="micro"),
        "f1": f1_score(y_true, y_pred, average="micro"),
    }

    # Generating a classification report.
    print(f'metrics:\n{metrics}')
    report_labels = list(label2id.values())
    report_target_names = list(label2id.keys())
    classification_report_text = classification_report(y_true, y_pred, labels=report_labels, target_names=report_target_names, digits=4, zero_division=0.0)
    classification_report_dict = classification_report(y_true, y_pred, labels=report_labels, target_names=report_target_names, digits=4, output_dict=True, zero_division=0.0)
    print(classification_report_text)

    return metrics, metrics, classification_report_dict

def calc_metrics_extraction(
    y_true: list[list[tuple[str, str]]], 
    y_pred: list[list[tuple[str, str]]], 
    texts: list[str], 
    verbose: bool = False
) -> tuple[dict, dict]:
    """
    Calculate span-based extraction metrics using MUC.
    
    :param y_true: list of gold spans, e.g., [[('PER', 'John Smith'), ...], ...]
    :param y_pred: list of predicted spans in the same format
    :param texts: list of original sentences
    :param verbose: whether to print per-example evaluation
    :return: dictionary with precision, recall, f1
    """
    results = muc.evaluate_all(y_pred, y_true, texts, verbose=verbose)
    for mode in results:
        del results[mode]["count"]

    csv_results = {}
    for mode in results:
        for metric in results[mode]:
            csv_results[f"{mode}_{metric}"] = results[mode][metric]
    return csv_results, results

def get_responses_classification_predictions(responses: list[dict], sc_runs) -> tuple[list, list, list]:
    predictions_list = []
    ratios_list = []
    hallucinated = []

    # Calculate score
    for i, resp_info in enumerate(responses):
        # Collect labels from all responses (Self-Consistency)
        runs_labels = []
        at_least_one_resp_with_no_hallucination = False
        for j, resp in enumerate(resp_info["responses"]):
            try:
                if resp and "label" in resp:
                    runs_labels.append(resp["label"])
                    at_least_one_resp_with_no_hallucination = True
                else:
                    # If the response is not in the expected format
                    print(f"Response index:{i}, {j} Unexpected response format: {resp}")
            except Exception as e:
                # If the response is not in the expected format, skip it
                print(f"Response index:{i}, {j} Error parsing response: {e}: Unexpected response format: {resp['parsed']}")
                continue

        # If no response was valid, skip this sentence.
        if not at_least_one_resp_with_no_hallucination:
            labels_with_ratio = None
        else:
            labels_with_ratio = self_consistency_classification(runs_labels, sc_runs)

        if labels_with_ratio:
            pred = max(labels_with_ratio.items(), key=lambda x: x[1])[0]
            ratio = labels_with_ratio[pred]
            hallucination = False
        else:
            pred = "EMPTY"
            ratio = None
            hallucination = True

        predictions_list.append(pred)
        ratios_list.append(ratio)
        hallucinated.append(hallucination)

    return predictions_list, ratios_list, hallucinated

def process_responses_classification(
    responses: list[dict], test: pd.DataFrame, label2id: dict, **kwargs
) -> tuple[dict, dict, DataFrame, dict]:
    """
    Calculate metrics for the given task after post-processing the results.
    :param results: results
    :param test: test data
    :param label2id: dictionary mapping labels to ids
    :return: metrics and test data after post-processing and the csv formatted metrics
    """
    if len(responses) != len(test):
        raise ValueError(f"Number of responses ({len(responses)}) does not match number of test examples ({len(test)})")

    sc_runs = kwargs["sc_runs"]
    predictions_list, ratios_list, hallucinated = get_responses_classification_predictions(responses, sc_runs)

    # Add idioms to test data and their ratios
    test["prediction"] = predictions_list
    test["ratio"] = ratios_list
    test["hallucinated"] = hallucinated
    print(f'\ntest in process_responses_classification: {test.head()}')
    # Calculate metrics for the current language
    sanitize_predictions(predictions_list, label2id)
    print(f'predictions_list after sanitize_predictions: {predictions_list}')

    csv_results, metrics, classification_report_dict = calc_metrics_classification(test["label"], predictions_list, label2id)
    # Add the hallucination number to the metrics
    metrics["hallucinations"] = int(test["hallucinated"].sum())
    csv_results["hallucinations"] = metrics["hallucinations"]

    return csv_results, metrics, test, classification_report_dict

def sanitize_predictions(predictions_list, label2id):
    valid_labels = list(label2id.keys())
    for i, pred in enumerate(predictions_list):
        if pred not in valid_labels:
            predictions_list[i] = "EMPTY"

def get_responses_extraction_predictions(responses: list[dict], pred_col = "prediction") -> tuple[list, list]:

    predictions = []
    hallucinated = []

    for i, resp_info in enumerate(responses):
        preds = []
        try:
            if isinstance(resp_info["responses"][0], dict):
                if pred_col in resp_info["responses"][0]:
                    preds = resp_info["responses"][0].get(pred_col, [])
                    hallucinated.append(False)
                else:
                    print(f"Response does not contain '{pred_col}' key: {resp_info['responses'][0]}")
                    preds = []
                    hallucinated.append(True)
            else:
                resp = resp_info["responses"][0]
                if resp:
                    if pred_col not in resp:
                        print("")
                        print(f"Response does not contain '{pred_col}' key: {resp}")
                        print(f"Response:{i}, marked as hallucination with original_resp:{resp}")
                        preds = []
                        hallucinated.append(True)
                    else:
                        preds = resp[pred_col]
                        hallucinated.append(False)
                else:
                    print("")
                    print(f"Unexpected response:{i}, marked as hallucination with original_resp:{resp}")
                    preds = []
                    hallucinated.append(True)
        except Exception as e:
            print("")
            print(f"Error parsing prediction with original_resp:{resp}, Error:{e}")
            preds = []
            hallucinated.append(True)
        predictions.append(preds)

    return predictions, hallucinated


def process_responses_extraction(
    responses: list[dict], 
    test: pd.DataFrame, 
    pred_col = "prediction",
    true_col = "ground_truth",
    text_col = "text",
    **kwargs
) -> tuple[dict, dict, DataFrame, None]:
    """
    Process extraction responses (span-based), calculate MUC metrics, and augment the test DataFrame.

    :param responses: List of model outputs, each as a dict with 'prediction' key holding extracted spans.
    :param test: The original test DataFrame containing the ground truths and text.
    :param pred_col: Column name for predictions (to be filled).
    :param true_col: Column name for true spans (ground truth).
    :param text_col: Column name for input text.
    :param mode: Evaluation mode for MUC metrics ('type', 'exact', 'partial', 'strict')
    :return: Tuple of (metrics dict, augmented DataFrame, None).
    """

    predictions, hallucinated = get_responses_extraction_predictions(responses, pred_col)
    # Normalize ground truths and make sure they are a list of tuples
    ground_truth = [
        [(tag, normalize_span(span)) for tag, span in row[true_col]]
        for _, row in test.iterrows()
    ]
    
    if text_col in test.columns:
        texts = [normalize_span(row[text_col]) for _, row in test.iterrows()]
    elif "passage" in test.columns:
        texts = [normalize_span(row["passage"]) for _, row in test.iterrows()]
    else:
        raise ValueError(f"Neither '{text_col}' nor 'passage' column found in test DataFrame.")
    # Normalize predictions
    try:
        predictions = [
            [(tag, normalize_span(span)) for tag, span in preds]
            for preds in predictions
        ]

    except Exception as e:
        print(f"Error normalizing predictions: {e}")
        for i, preds in enumerate(predictions):
            if isinstance(preds, str):
                print(f"Prediction at index {i} is a string: {preds}")
                # Convert string to empty list
                predictions[i] = []
            elif isinstance(preds, list):
                new_preds = []
                for j, pred in enumerate(preds):
                    if isinstance(pred, str):
                        print(f"Prediction at index {i}, element {j} is a string: {pred}")
                    elif len(pred) != 2 or not isinstance(pred[0], str) or not isinstance(pred[1], str):
                        print(f"Prediction at index {i}, element {j} is not a valid tuple: {pred}")
                    else:
                        new_preds.append((pred[0], normalize_span(pred[1])))
                predictions[i] = new_preds
            else:
                print(f"Prediction at index {i} is not a string or list: {preds}")
                predictions[i] = []
        
    # Compute metrics
    csv_results, metrics = calc_metrics_extraction(ground_truth, predictions, texts, verbose=False)
    metrics["hallucinations"] = int(sum(hallucinated))
    csv_results["hallucinations"] = metrics["hallucinations"]

    # Add predictions and hallucination flags to DataFrame
    test[pred_col] = predictions
    test["hallucinated"] = hallucinated

    return csv_results, metrics, test, None

def normalize_span(span):
    if span is None:
        return None
    span = span.lower().replace("-", "")
    return span

def send_email(
    config_path: str = "mail_config.yaml",
    app_password: str = "",
    sender_email: str = "",
    receiver_email: str = "",
    subject: str = "Python Script Finished",
    body: str = "Hey, your Python script just finished running.",
) -> int:
    """
    Send an email notification using SMTP.
    If config_path is provided, it will load the sender's email and app password from the config file.
    If no receiver email is provided, it defaults to the sender's email.
    :param sender_email: Sender's email address
    :param receiver_email: Receiver's email address
    :param app_password: App password for the sender's email account
    :param subject: Subject of the email
    :param body: Body of the email
    :return: 1 if successful, 0 if failed
    """
    if app_password == "" or sender_email == "":
        # Make sure the config file exists
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                sender_email = config.get("sender_email")
                app_password = config.get("app_password")
        except FileNotFoundError:
            print(f"❌ Config file '{config_path}' not found.")
            return 0

    if receiver_email == "":
        receiver_email = sender_email

    # Create the email message
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ Email sent successfully.")
        return 1
    except Exception as e:
        print("❌ Failed to send email:", e)
        return 0


####################################################################################################
