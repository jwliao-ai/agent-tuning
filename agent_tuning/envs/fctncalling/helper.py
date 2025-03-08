import re
import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import Union


from agent_tuning.envs.fctncalling._apply_function_credential_config import apply_function_credential_config
from agent_tuning.envs.fctncalling.constant import *
from agent_tuning.envs.fctncalling.checker.constant import *
from agent_tuning.envs.fctncalling.checker.executable.custom_exception import BadAPIStatusError


def extract_category(input_string: Union[str, Path]) -> str:
    input_string = str(input_string)
    pattern = fr".*{VERSION_PREFIX}_(\w+?)(?:_score|_result)?\.json"
    match = re.search(pattern, input_string)

    # Check if there's a match and extract the captured group
    if match:
        return match.group(1)  # the first captured group (\w+)
    else:
        raise ValueError(f"Could not extract the test category from the input string: {input_string}")

def find_file_with_suffix(folder_path: Path, suffix: str) -> Path:
    for json_file in folder_path.glob("*.json"):
        if extract_category(json_file) == suffix:
            return json_file
    raise FileNotFoundError(f"No JSON file found with suffix: {suffix}")

def is_multi_turn(category):
    return "multi_turn" in category

def contain_multi_turn_irrelevance(category):
    return "miss_func" in category or "miss_param" in category

def is_executable(category):
    return "exec" in category or "rest" in category

def is_rest(category):
    return "rest" in category

def is_relevance_or_irrelevance(category):
    return "relevance" in category or "irrelevance" in category

def is_chatable(category):
    return "chatable" in category

def is_java(category):
    return "java" in category

def is_js(category):
    return "javascript" in category

def is_sql(category):
    return "sql" in category

def load_file(file_path):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result

def write_list_of_dicts_to_file(filename, data, subdir=None):
    if subdir:
        # Ensure the subdirectory exists
        os.makedirs(subdir, exist_ok=True)

        # Construct the full path to the file
        filename = os.path.join(subdir, filename)

    # Write the list of dictionaries to the file in JSON format
    with open(filename, "w") as f:
        for i, entry in enumerate(data):
            # Go through each key-value pair in the dictionary to make sure the values are JSON serializable
            entry = make_json_serializable(entry)
            json_str = json.dumps(entry)
            f.write(json_str)
            if i < len(data) - 1:
                f.write("\n")


def make_json_serializable(value):
    if isinstance(value, dict):
        # If the value is a dictionary, we need to go through each key-value pair recursively
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        # If the value is a list, we need to process each element recursively
        return [make_json_serializable(item) for item in value]
    else:
        # Try to serialize the value directly, and if it fails, convert it to a string
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

def is_function_calling_format_output(decoded_output):
    # Ensure the output is a list of dictionaries
    if type(decoded_output) == list:
        for item in decoded_output:
            if type(item) != dict:
                return False
        return True
    return False

def is_executable_format_output(decoded_output):
    # Ensure the output is a list of strings (one or more strings)
    if type(decoded_output) == list:
        if len(decoded_output) == 0:
            return False
        for item in decoded_output:
            if type(item) != str:
                return False
        return True
    return False

def is_rest_format_output(decoded_output):
    # Ensure the output is a list of one string
    if type(decoded_output) == list:
        if len(decoded_output) == 1 and type(decoded_output[0]) == str:
            return True
    return False

def is_empty_output(decoded_output):
    # This function is a patch to the ast decoder for relevance detection
    # Sometimes the ast decoder will parse successfully, but the input doens't really have a function call
    # [], [{}], and anything that is not in function calling format is considered empty (and thus should be marked as correct)
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False

def api_status_sanity_check_rest():

    # We only need to import the executable_checker_rest in this function. So a local import is used.
    from fctncalling_rft.envs.fctncalling.checker.executable.executable_checker import executable_checker_rest

    ground_truth_dummy = load_file(REST_API_GROUND_TRUTH_FILE_PATH)

    # Use the ground truth data to make sure the API is working correctly
    apply_function_credential_config(input_path=REST_API_GROUND_TRUTH_FILE_PATH)

    ground_truth_replaced = load_file(REST_API_GROUND_TRUTH_FILE_PATH)
    write_list_of_dicts_to_file(REST_API_GROUND_TRUTH_FILE_PATH, ground_truth_dummy)

    correct_count = 0
    errors = []
    for idx, data in tqdm(
        enumerate(ground_truth_replaced),
        total=len(ground_truth_replaced),
        desc="API Status Test (REST)",
    ):
        status = executable_checker_rest(data["ground_truth"], idx)
        if status["valid"]:
            correct_count += 1
        else:
            errors.append((data, status))

    if correct_count != len(ground_truth_replaced):
        raise BadAPIStatusError(errors, f"{len(ground_truth_replaced) - correct_count} / {len(ground_truth_replaced)}")


def api_status_sanity_check_executable():
    from fctncalling_rft.envs.fctncalling.checker.executable.executable_checker import executable_checker_simple

    ground_truth = load_file(EXECTUABLE_API_GROUND_TRUTH_FILE_PATH)
    correct_count = 0
    errors = []
    for data in tqdm(
        ground_truth, total=len(ground_truth), desc="API Status Test (Non-REST)"
    ):
        status = executable_checker_simple(
            data["ground_truth"][0],
            data["execution_result"][0],
            data["execution_result_type"][0],
            True,
        )
        if status["valid"]:
            correct_count += 1
        else:
            errors.append((data, status))

    if correct_count != len(ground_truth):
        raise BadAPIStatusError(errors, f"{len(ground_truth) - correct_count} / {len(ground_truth)}")


def display_api_status_error(rest_error, executable_error, display_success=False):
    if not rest_error and not executable_error:
        if display_success:
            print("🟢 All API Status Test Passed!")
        return None
    
    print(f"\n{RED_FONT}{'-' * 18} Executable Categories' Error Bounds Based on API Health Status {'-' * 18}{RESET}\n")

    if rest_error:
        print(f"❗️ Warning: Unable to verify health of executable APIs used in executable test category (REST). Please contact API provider.\n")
        print(f"{rest_error.error_rate} APIs affected:\n")
        for data, status in rest_error.errors:
            print(f"  - Test Case: {data['ground_truth']}")
            print(f"    Error Type: {status['error_type']}\n")

    if executable_error:
        print(f"❗️ Warning: Unable to verify health of executable APIs used in executable test categories (Non-REST). Please contact API provider.\n")
        print(f"{executable_error.error_rate} APIs affected:\n")
        for data, status in executable_error.errors:
            print(f"  - Test Case: {data['ground_truth'][0]}")
            print(f"    Error Type: {status['error_type']}\n")

    print(f"{RED_FONT}{'-' * 100}\n{RESET}")

def get_executable_expected_output(prompt_file_path):
    # Before we run the evaluation, we need to add the "execution_result" field to the prompt file, using the ground truth data.
    prompt_content = load_file(prompt_file_path)
    exec_dict = {}
    for item in tqdm(prompt_content, desc="Getting Executable Expected Output"):
        execution_result = []
        ground_truth = item["ground_truth"]
        for i in range(len(ground_truth)):
            exec(
                "from fctncalling_rft.envs.fctncalling.checker.executable.data.executable_python_function import *"
                + "\nresult="
                + ground_truth[i],
                exec_dict,
            )
            execution_result.append(exec_dict["result"])
        item["execution_result"] = execution_result

    write_list_of_dicts_to_file(prompt_file_path, prompt_content)


def get_executable_expected_output(ground_truth):
    # Before we run the evaluation, we need to get the executable expected output, using the ground truth data.
    exec_dict = {}
    execution_result = []
    for i in range(len(ground_truth)):
        exec(
            "from fctncalling_rft.envs.fctncalling.checker.executable.data.executable_python_function import *"
            + "\nresult="
            + ground_truth[i],
            exec_dict,
        )
        execution_result.append(exec_dict["result"])
    return execution_result

def load_func_docs(involved_classes: list[str], func_doc_path: str, missed_function: dict):
    functions = []
    holdout_functions = {}
    for class_name in involved_classes:
        functions.extend(load_file(func_doc_path + MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]))
    for turn_index, missed_func_names in missed_function.items():
        holdout_functions[turn_index] = []
        for missed_func_name in missed_func_names:
            for i, func_doc in enumerate(functions):
                if func_doc["name"] == missed_func_name:
                    # Add the missed function doc to the missed_function list
                    holdout_functions[turn_index].append(func_doc)
                    # Remove it from the function list
                    functions.pop(i)
                    break
    return functions, holdout_functions