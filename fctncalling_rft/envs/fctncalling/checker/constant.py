### CONSTANT for model handler ###

MAXIMUM_ROUND_LIMIT = 20

DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
"""

DEFAULT_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + """
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
)

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = "I have updated some more functions you can choose from. What about now?"

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING = "{functions}\n" + DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC

GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

GORILLA_TO_PYTHON = {
    "integer": "int",
    "number": "float",
    "float": "float",
    "string": "str",
    "boolean": "bool",
    "bool": "bool",
    "array": "list",
    "list": "list",
    "dict": "dict",
    "object": "dict",
    "tuple": "tuple",
    "any": "str",
    "byte": "int",
    "short": "int",
    "long": "int",
    "double": "float",
    "char": "str",
    "ArrayList": "list",
    "Array": "list",
    "HashMap": "dict",
    "Hashtable": "dict",
    "Queue": "list",
    "Stack": "list",
    "Any": "str",
    "String": "str",
    "Bigint": "int",
}


JAVA_TYPE_CONVERSION = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,  # this can be `queue.Queue` as well, for simplicity we check with list
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}

UNDERSCORE_TO_DOT = [
    "gpt-4o-2024-08-06-FC",
    "gpt-4o-2024-05-13-FC",
    "gpt-4o-mini-2024-07-18-FC",
    "gpt-4-turbo-2024-04-09-FC",
    "gpt-4-1106-preview-FC",
    "gpt-4-0125-preview-FC",
    "gpt-4-0613-FC",
    "gpt-3.5-turbo-0125-FC",
    "claude-3-opus-20240229-FC",
    "claude-3-sonnet-20240229-FC",
    "claude-3-haiku-20240307-FC",
    "claude-3-5-sonnet-20240620-FC",
    "open-mistral-nemo-2407-FC",
    "open-mixtral-8x22b-FC",
    "mistral-large-2407-FC",
    "mistral-large-2407-FC",
    "mistral-small-2402-FC",
    "mistral-small-2402-FC",
    "gemini-1.5-pro-002-FC",
    "gemini-1.5-pro-001-FC",
    "gemini-1.5-flash-002-FC",
    "gemini-1.5-flash-001-FC",
    "gemini-1.0-pro-002-FC",
    "meetkai/functionary-small-v3.1-FC",
    "meetkai/functionary-small-v3.2-FC",
    "meetkai/functionary-medium-v3.1-FC",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-2-Pro-Llama-3-70B",
    "NousResearch/Hermes-2-Pro-Mistral-7B",
    "NousResearch/Hermes-2-Theta-Llama-3-8B",
    "NousResearch/Hermes-2-Theta-Llama-3-70B",
    "command-r-plus-FC",
    "command-r-plus-FC-optimized",
    "THUDM/glm-4-9b-chat",
    "ibm-granite/granite-20b-functioncalling",
    "yi-large-fc",
]


### CONSTANT for checker ###

from pathlib import Path

REAL_TIME_MATCH_ALLOWED_DIFFERENCE = 0.2

# These two files are for the API status sanity check
REST_API_GROUND_TRUTH_FILE_PATH = (
    "./executable/data/api_status_check_ground_truth_REST.json"
)
EXECTUABLE_API_GROUND_TRUTH_FILE_PATH = (
    "./executable/data/api_status_check_ground_truth_executable.json"
)

# This is the ground truth file for the `rest` test category
REST_EVAL_GROUND_TRUTH_PATH = "./executable/data/rest-eval-response_v5.jsonl"

COLUMNS_NON_LIVE = [
    "Rank",
    "Model",
    "Non_Live Overall Acc",
    "AST Summary",
    "Exec Summary",
    "Simple AST",
    "Python Simple AST",
    "Java Simple AST",
    "JavaScript Simple AST",
    "Multiple AST",
    "Parallel AST",
    "Parallel Multiple AST",
    "Simple Exec",
    "Python Simple Exec",
    "REST Simple Exec",
    "Multiple Exec",
    "Parallel Exec",
    "Parallel Multiple Exec",
    "Irrelevance Detection",
]


COLUMNS_LIVE = [
    "Rank",
    "Model",
    "Live Overall Acc",
    "AST Summary",
    "Python Simple AST",
    "Python Multiple AST",
    "Python Parallel AST",
    "Python Parallel Multiple AST",
    "Irrelevance Detection",
    "Relevance Detection",
]


COLUMNS_MULTI_TURN = [
    "Rank",
    "Model",
    "Multi Turn Overall Acc",
    "Base",
    "Miss Func",
    "Miss Param",
    "Long Context",
    "Composite",
]


COLUMNS_OVERALL = [
    "Rank",
    "Overall Acc",
    "Model",
    "Model Link",
    "Cost ($ Per 1k Function Calls)",
    "Latency Mean (s)",
    "Latency Standard Deviation (s)",
    "Latency 95th Percentile (s)",
    "Non-Live AST Acc",
    "Non-Live Simple AST",
    "Non-Live Multiple AST",
    "Non-Live Parallel AST",
    "Non-Live Parallel Multiple AST",
    "Non-Live Exec Acc",
    "Non-Live Simple Exec",
    "Non-Live Multiple Exec",
    "Non-Live Parallel Exec",
    "Non-Live Parallel Multiple Exec",
    "Live Acc",
    "Live Simple AST",
    "Live Multiple AST",
    "Live Parallel AST",
    "Live Parallel Multiple AST",
    "Multi Turn Acc",
    "Multi Turn Base",
    "Multi Turn Miss Func",
    "Multi Turn Miss Param",
    "Multi Turn Long Context",
    "Multi Turn Composite",
    "Relevance Detection",
    "Irrelevance Detection",
    "Organization",
    "License",
]


# Price got from AZure, 22.032 per hour for 8 V100, Pay As You Go Total Price
# Reference: https://azure.microsoft.com/en-us/pricing/details/machine-learning/
V100_x8_PRICE_PER_HOUR = 22.032

RED_FONT = "\033[91m"
RESET = "\033[0m"

# Construct the full path for other modules to use
script_dir = Path(__file__).parent
REST_API_GROUND_TRUTH_FILE_PATH = (script_dir / REST_API_GROUND_TRUTH_FILE_PATH).resolve()
EXECTUABLE_API_GROUND_TRUTH_FILE_PATH = (script_dir / EXECTUABLE_API_GROUND_TRUTH_FILE_PATH).resolve()
REST_EVAL_GROUND_TRUTH_PATH = (script_dir / REST_EVAL_GROUND_TRUTH_PATH).resolve()


### CONSTANT for others ###

from pathlib import Path

# NOTE: These paths are relative to the `bfcl` directory where this script is located.
RESULT_PATH = "../result/"
PROMPT_PATH = "../data/"
POSSIBLE_ANSWER_PATH = "../data/possible_answer/"
SCORE_PATH = "../score/"
DOTENV_PATH = "../.env"

VERSION_PREFIX = "BFCL_v3"

# These are in the PROMPT_PATH
TEST_FILE_MAPPING = {
    "exec_simple": f"{VERSION_PREFIX}_exec_simple.json",
    "exec_parallel": f"{VERSION_PREFIX}_exec_parallel.json",
    "exec_multiple": f"{VERSION_PREFIX}_exec_multiple.json",
    "exec_parallel_multiple": f"{VERSION_PREFIX}_exec_parallel_multiple.json",
    "simple": f"{VERSION_PREFIX}_simple.json",
    "irrelevance": f"{VERSION_PREFIX}_irrelevance.json",
    "parallel": f"{VERSION_PREFIX}_parallel.json",
    "multiple": f"{VERSION_PREFIX}_multiple.json",
    "parallel_multiple": f"{VERSION_PREFIX}_parallel_multiple.json",
    "java": f"{VERSION_PREFIX}_java.json",
    "javascript": f"{VERSION_PREFIX}_javascript.json",
    "rest": f"{VERSION_PREFIX}_rest.json",
    "sql": f"{VERSION_PREFIX}_sql.json",
    "chatable": f"{VERSION_PREFIX}_chatable.json",
    # Live Datasets
    "live_simple": f"{VERSION_PREFIX}_live_simple.json",
    "live_multiple": f"{VERSION_PREFIX}_live_multiple.json",
    "live_parallel": f"{VERSION_PREFIX}_live_parallel.json",
    "live_parallel_multiple": f"{VERSION_PREFIX}_live_parallel_multiple.json",
    "live_irrelevance": f"{VERSION_PREFIX}_live_irrelevance.json",
    "live_relevance": f"{VERSION_PREFIX}_live_relevance.json",
    # Multi-turn Datasets
    "multi_turn_base": f"{VERSION_PREFIX}_multi_turn_base.json",
    "multi_turn_miss_func": f"{VERSION_PREFIX}_multi_turn_miss_func.json",
    "multi_turn_miss_param": f"{VERSION_PREFIX}_multi_turn_miss_param.json",
    "multi_turn_long_context": f"{VERSION_PREFIX}_multi_turn_long_context.json",
    "multi_turn_composite": f"{VERSION_PREFIX}_multi_turn_composite.json",
}

TEST_COLLECTION_MAPPING = {
    "all": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
        # "multi_turn_composite",  # Composite is currently not included in the leaderboard
    ],
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
        # "multi_turn_composite",  # Composite is currently not included in the leaderboard
    ],
    "single_turn": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "live": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "non_live": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
    ],
    # TODO: Update this mapping
    "ast": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "executable": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "rest",
    ],
    "non_python": [
        "java",
        "javascript",
    ],
    "python": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "python_ast": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
}


# Construct the full path to use by other scripts
script_dir = Path(__file__).parent
RESULT_PATH = (script_dir / RESULT_PATH).resolve()
PROMPT_PATH = (script_dir / PROMPT_PATH).resolve()
POSSIBLE_ANSWER_PATH = (script_dir / POSSIBLE_ANSWER_PATH).resolve()
SCORE_PATH = (script_dir / SCORE_PATH).resolve()
DOTENV_PATH = (script_dir / DOTENV_PATH).resolve()

RESULT_PATH.mkdir(parents=True, exist_ok=True)
SCORE_PATH.mkdir(parents=True, exist_ok=True)
