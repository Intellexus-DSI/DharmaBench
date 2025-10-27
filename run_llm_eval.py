"""
Running an in-context-learning evaluation of an LLM on a set of downstream tasks.
Based on config_eval_llm.yaml.
"""

import os
import yaml
import json
import argparse
from argparse import Namespace
import pandas as pd
from transformers import set_seed
import sys
from langchain.schema import AIMessage  # or wherever AIMessage is from


from utils.utils import (
    MERGE_COLUMNS,
    get_logger,
    set_keys,
    parse_response,
    send_email,
    run_batch,
    run_individual,
)
from utils.models import get_model, get_schema

from tasks.MCS import MCS
from tasks.SMDS import SMDS
from tasks.QUDS import QUDS
from tasks.AACT import AACT
from tasks.VPCT import VPCT
from tasks.SCCT import SCCT
from tasks.THCT import THCT
from tasks.QUDT import QUDT
from tasks.SDT import SDT
from tasks.RCMS_RSMT import RCMS_RSMT
from tasks.VPCS import VPCS
from tasks.RCDS import RCDS

####################################################################################################
# Command-line arguments

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="config_llm_eval.yaml",
    help="Path to the config file",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument(
    "--sc_runs",
    type=int,
    default=None,
    help="Number of self-consistency runs",
)
parser.add_argument(
    "--responses_dir",
    type=str,
    default=None,
    help="Directory with responses",
)

DETECTION_TASKS = [
    "QUDS",
    "SDT",
    "SMDS",
    "QUDT",
    "RCDS",
]

#####################################################################################################


####################################################################################################
# Define the utils for each task
def get_tasks(task: str) -> dict:
    """
    Get utils for the given task.
    :param task: task name
    :return: utils
    """
    if task == "MCS":
        return MCS
    elif task == "QUDS":
        return QUDS
    elif task == "SDT":
        return SDT
    elif task == "SMDS":
        return SMDS
    elif task == "AACT":
        return AACT
    elif task == "THCT":
        return THCT
    elif task == "VPCT":
        return VPCT
    elif task == "QUDT":
        return QUDT
    elif task == "SCCT":
        return SCCT
    elif "RCMS" in task:
        return RCMS_RSMT
    elif "RCMT" in task:
        return RCMS_RSMT
    elif task == "RCDS":
        return RCDS
    elif task == "VPCS":
        return VPCS
    else:
        raise ValueError(f"Task {task} is not supported")


#####################################################################################################


######################################################################################################
# Main function
def main():
    # Get logger
    logger = get_logger(__name__)

    # Get CMD args
    cmd_args = parser.parse_args()
    logger.info(f"CMD args: {cmd_args}")

    # Load keys
    with open("./keys.yaml", "r") as f:
        keys = yaml.safe_load(f)
    logger.info("Loaded API keys")
    # Set keys
    set_keys(keys)

    # Load config
    config_file = cmd_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")

    logger.info(f"Started evaluation set on language: {config['task']}")
    if "responses_dir" in cmd_args and cmd_args.responses_dir:
        config["responses_dir"] = cmd_args.responses_dir
        logger.info(f"Updated config with responses_dir: {config['responses_dir']}")

    # Check if responses were given
    if config["responses_dir"]:
        responses_dir = config["responses_dir"]
        # Get the original config
        with open(os.path.join(config["responses_dir"], "config.yaml"), "r") as f:
            orig_config = yaml.safe_load(f)
        # Update config with the original config
        config.update(orig_config)
        # Update responses_dir to the original one
        config["responses_dir"] = responses_dir
        logger.info(f"Updated config from {config['responses_dir']}")

    # Update config with CMD args
    for key, value in vars(cmd_args).items():
        # Skip None values
        if value is None:
            continue
        config[key] = value

    # Set seed
    set_seed(config["seed"])

    # Get experiment name
    model_name = config["model"].split("/")[-1]
    exp_name = f"{config['task']}_{model_name}_{config['prompt_type']}_shots_{config['shots']}_sc{config['sc_runs']}_tmp{config['temperature']}_seed{config['seed']}"

    # Get utils
    tasks = Namespace(**get_tasks(config["task"]))

    # Get data_dir
    data_dir = config["data_dir"]

    # Assert
    if "few" in config["prompt_type"]:
        assert config["shots"] > 0, "Shots must be greater than 0"
        assert config["shots"] % 2 == 0, "Shots must be even"
    if "zero" in config["prompt_type"]:
        assert config["shots"] == 0, "Shots must be 0 for zero-shot"
    if config["sc_runs"] > 1:
        assert (
            config["temperature"] == 0.8
        ), "Temperature must be 0.8 for self-consistency"

    # Create experiment directory
    config["logs_dir"] = os.path.join(config["logs_dir"], config["task"])
    exp_dir = os.path.join(config["logs_dir"], exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Write config to file
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load data
    _, test = tasks.get_data(data_dir=data_dir, task=config["task"])
    logger.info(f"Loaded data: test shape={test.shape}")

    # Cut data for debugging
    if config["debug"]:
        if "debug_samples" in config and config.get("debug_samples") is not None:
            # Choose those samples from the test set
            test = test.iloc[config["debug_samples"]]
        else:
            test = test.sample(n=config["num_samples"], random_state=config["seed"])
            
    # Check if responses were given
    if config["responses_dir"]:
        responses_file = os.path.join(config["responses_dir"], "responses.json")
        with open(responses_file, "r", encoding="utf-8-sig") as f:
            responses = json.load(f)
        logger.info(f"Loaded responses: {len(responses)}")
    else:
        kwargs = {}

        # Add optional parameters if they exist
        if "requests_per_second" in config:
            kwargs["requests_per_second"] = config["requests_per_second"]
        if "check_every_n_seconds" in config:
            kwargs["check_every_n_seconds"] = config["check_every_n_seconds"]

        llm = get_model(config["model"], config["temperature"], config["use_rate_limiter"], **kwargs)

        task_type = "detection" if config["task"] in DETECTION_TASKS else "classification"
        # Get schema for the task
        schema = get_schema(model_name=config["model"], task_type=task_type)
        if schema is not None:
            # Apply structured output if required
            llm = llm.with_structured_output(schema, include_raw=True)
            
        prompt = tasks.get_prompt(config=config)
        logger.info(f'prompt in run_llm_eval: {prompt}\n\n')
        chain = prompt | llm
        logger.info(f'chain in run_llm_eval: {chain}\n\n')
        user_inputs = tasks.get_user_inputs(test)
        logger.info(f'user_input in run_llm_eval for example: {user_inputs[0]}\n\n')
        # Initialize responses
        # Filter columns_to_log to only include columns that exist in the test DataFrame
        valid_columns = [col for col in tasks.columns_to_log if col in test.columns]
        if len(valid_columns) < len(tasks.columns_to_log):
            missing_cols = set(tasks.columns_to_log) - set(valid_columns)
            logger.warning(f"Some columns from columns_to_log are missing in test data: {missing_cols}")
        
        responses = [
            {**{key: row[key] for key in valid_columns}, "responses": [], "raw_response": []}
            for _, row in test.iterrows()
        ]

        # Run multiple runs for self-consistency
        logger.info(
            f"Running {'batch' if config['batched'] else 'individual'} runs for {config['sc_runs']} iterations"
        )

        for run_index in range(config["sc_runs"]):
            logger.info(f"Starting run {run_index + 1}/{config['sc_runs']}")

            if config["batched"]:
                raw_responses_run = run_batch(chain, user_inputs, run_index, logger)
            else:
                raw_responses_run = run_individual(
                    chain, user_inputs, run_index, exp_dir, logger
                )

            # Parse and save responses
            for i, resp in enumerate(raw_responses_run):

                if "stop_on_response_error" in config and config["stop_on_response_error"]:
                    if resp is None:
                        print(f"No response received for run_index:[{run_index}], i:[{i}] stopping evaluation!!!")
                        if sys.platform == "darwin":
                            os.system(f'say "No response received for run_index:[{run_index}], i:[{i}] stopping evaluation!!!"')
                        sys.exit(42)

                print(resp)
                # Add a raw response to the response dict.
                if hasattr(resp, 'content'):
                    raw_response = resp.content
                elif isinstance(resp, AIMessage):
                    raw_response = resp.content
                    
                elif isinstance(resp, dict):
                    if "raw" in resp:
                        raw_response = resp["raw"]
                        raw_response = raw_response.content if hasattr(raw_response, 'content') else raw_response
                        if type(raw_response) is not str:
                            raw_response = str(raw_response)
                else:
                    raw_response = str(resp)
                responses[i]["raw_response"].append(raw_response) # resp.content if hasattr(resp, 'content') else resp)
                try:    
                    # Parse response and append to responses
                    responses[i]["responses"].append(parse_response(resp))
                except Exception as e:
                    logger.error(f"Error parsing response: {e}, response: {resp}")
                    responses[i]["responses"].append({})

    # Save responses to a file
    print(responses[0])
    with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
        json.dump(responses, f, indent=1, ensure_ascii=False)
    logger.info(f"Saved responses to {exp_dir}")

    # Process responses and calculate metrics
    try:
        csv_results, metrics, test, classification_report_dict = tasks.process_responses(
            responses,
            test,
            config=config,
            label2id=tasks.label2id,
            sc_runs=config["sc_runs"],
        )
    except Exception as e:
        raise RuntimeError(f"Error processing responses: {e}")

    logger.info(f"Metrics: {metrics}")
    # Write metrics to file
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if classification_report_dict:
        with open(os.path.join(exp_dir, "classification_report.json"), 'w', encoding='utf-8') as f:
            json.dump(classification_report_dict, f, indent=2, ensure_ascii=False)

    # Save results to file
    if "text" in test:
        test["text"] = test["text"].str.replace("\n", "\\n", regex=False)
    test.to_csv(os.path.join(exp_dir, "results.tsv"), index=False, sep="\t")
    logger.info(f"Saved results to {exp_dir}")

    run_res_row = {
        **{key: config[key] for key in config if key in MERGE_COLUMNS},
        **csv_results,
    }
    # Add debug
    run_res_row["debug"] = config["debug"]
    run_res = pd.DataFrame([run_res_row])

    # Load task results
    task_res_dir = os.path.join(config["results_dir"], config["task"])
    os.makedirs(task_res_dir, exist_ok=True)

    task_res_file = os.path.join(task_res_dir, "all_results.csv")
    if os.path.exists(task_res_file):
        task_res = pd.read_csv(task_res_file)
    else:
        task_res = pd.DataFrame(columns=MERGE_COLUMNS)
        task_res.to_csv(task_res_file, index=False)

    # Merge results with task results based on merge columns
    merged = task_res.merge(
        run_res[MERGE_COLUMNS], on=MERGE_COLUMNS, how="left", indicator=True
    )
    updated_df = task_res[merged["_merge"] == "left_only"].copy()

    # adding timestamp
    run_res['timestamp'] = pd.Timestamp.now()

    # Append new row
    task_res = pd.concat([updated_df, run_res], ignore_index=True)

    task_res.to_csv(task_res_file, index=False)
    logger.info(f"Saved all results to {task_res_file}")


###################################################################################
# Main
if __name__ == "__main__":
    main()

    os.system('say "your downstream task has finished"')

    # Send email notification
    # send_email()
