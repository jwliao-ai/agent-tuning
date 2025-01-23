import json
import numpy as np
import random
from pathlib import Path
from fctncalling_rft.envs.fctncalling.utils import *
from fctncalling_rft.envs.fctncalling.helper import *
from fctncalling_rft.envs.fctncalling.handler.hammer import HammerHandler
from fctncalling_rft.envs.fctncalling.checker.ast.ast_checker import ast_checker
from fctncalling_rft.envs.fctncalling.checker.executable.executable_checker import executable_checker_rest, executable_checker_non_rest
from fctncalling_rft.envs.fctncalling.checker.multi_turn.multi_turn_checker import multi_turn_checker, multi_turn_irrelevance_checker
from fctncalling_rft.envs.fctncalling.checker.multi_turn.multi_turn_utils import *
from fctncalling_rft.envs.fctncalling.checker.executable.custom_exception import BadAPIStatusError, NoAPIKeyError

class FctnCallingEnv:

    def __init__(self, rank, model_name, num_agents, dataset_path, no_nan=True):
        self.rank = rank
        self.no_nan = no_nan
        self.n_agents = num_agents
        self.model_name = model_name
        self.handler = HammerHandler(model_name=model_name)
        self.api_sanity_check = False
        self.dataset = []
        self.entry_idx = 0
        self.max_steps = 5
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        current_file_dir = Path(__file__).resolve().parent
        self.func_doc_path = current_file_dir / "multi_turn_func_doc"
        
        # A flag to indicate if the API has been tested.
        # We should always test the API with ground truth first before running the executable tests.
        # Sometimes the API may not be working as expected and we want to catch that before running the evaluation to ensure the results are accurate.
        self.API_TESTED = False
        self.API_STATUS_ERROR_REST = None
        self.API_STATUS_ERROR_EXECUTABLE = None

        # self.log_file = f"../../mat/envs/fctncalling/cache/{flag}_{rank}.py"
        # with open(self.log_file, "w") as f:
        #     f.write("")

    def reset(self):
        self.entry: dict = random.choice(self.dataset)
        self.id: str = self.entry["id"]
        self.category = self.id.rsplit("_", 1)[0]
        self.id = self.category + "_" + str(self.entry_idx)
        self.entry["id"] = self.id
        self.entry_idx += 1
        self.question: list[list[dict]] = self.entry["question"]
        if not is_relevance_or_irrelevance(self.category):
            self.ground_truth = self.entry["ground_truth"]  # list[dict] for single-turn or list[list[str]] for multi-turn)

        if is_chatable(self.category) or is_sql(self.category):  # not support now
            self.reset()

        if is_multi_turn(self.category):
            self.task_progress = 1
            self.steps_in_turn = 1
            self.func_calls_in_turns = [[] for _ in range(len(self.question))]
            self.initial_config = self.entry['initial_config']
            self.involved_classes = self.entry['involved_classes']
            self.function, self.holdout_function = load_func_docs(self.involved_classes, self.func_doc_path, self.entry.get("missed_function", {}))
            self.entry['function'] = self.function
            self.entry['holdout_function'] = self.holdout_function
        else:
            self.function: list[dict] = self.entry["function"]

        self.language = "Python"
        if is_java(self.category):
            self.language = "Java"
        if is_js(self.category):
            self.language = "JavaScript"

        # get executable expected output (depending on API status, not support now)
        if is_executable(self.category):
            # We only test the API with ground truth once
            if not self.API_TESTED and self.api_sanity_check:
                print("---- Sanity checking API status ----")
                try:
                    api_status_sanity_check_rest()
                except BadAPIStatusError as e:
                    self.API_STATUS_ERROR_REST = e

                try:
                    api_status_sanity_check_executable()
                except BadAPIStatusError as e:
                    self.API_STATUS_ERROR_EXECUTABLE = e

                display_api_status_error(
                    self.API_STATUS_ERROR_REST,
                    self.API_STATUS_ERROR_EXECUTABLE,
                    display_success=True,
                )
                print("Continuing evaluation...")
                self.API_TESTED = True

            if not is_rest(self.category):
                self.entry["execution_result"] = get_executable_expected_output(self.ground_truth)

        self.history = self.handler._pre_query_processing_prompting(self.entry)
        self.history = self.handler.add_first_turn_message_prompting(self.history, self.question[0])
        formatted_prompt = self.handler.format_prompt(self.history["message"], self.history["function"])

        obs = np.array([formatted_prompt for _ in range(self.n_agents)], dtype=np.object_)

        return obs

    def step(self, action):
        action = action[-1]
        self.history["message"].extend([{"role": "assistant", "content": action}])

        results = []
        if is_relevance_or_irrelevance(self.category):
            relevance_result = self.relevance_or_irrelevance_check(action)
            results.append(relevance_result)
        elif is_executable(self.category):
            executable_result = self.executable_check(action)
            results.append(executable_result)
        elif is_multi_turn(self.category):
            multi_turn_result = self.multi_turn_check(action)
            results.append(multi_turn_result)
        else:
            ast_result = self.ast_check(action)
            results.append(ast_result)

        next_obs, rewards, dones = self.transition_and_reward(results)

        infos = [{"history": self.history, "traj_length": self.task_progress} for _ in range(self.n_agents)]
        return next_obs, rewards, dones, infos

    def render(self, **kwargs):
        # self.env.render(**kwargs)
        pass

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)

    def get_env_info(self):

        env_info = {"n_agents": self.n_agents}
        return env_info

    def relevance_or_irrelevance_check(self, action) -> dict:
        decode_error = None
        try:
            decoded_action = self.handler.decode_ast(action, self.language)
            contain_func_call = False if is_empty_output(decoded_action) else True
        except Exception as e:
            contain_func_call = False
            decode_error = str(e)
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": False,
                "error": [f"Failed to decode ast. {decode_error}"],
                "error_type": "ast_decoder:decoder_failed",
                "prompt": self.entry,
                "action_raw": action,
            }
        success = (
            not contain_func_call
            if "irrelevance" in self.category
            else contain_func_call
        )
        result = {
            "id": self.id,
            "model_name": self.model_name,
            "test_category": self.category,
            "valid": success,
            "prompt": self.entry,
            "action_raw": action,
            "action_decoded": decoded_action,
        }
        if not success:
            if "irrelevance" in self.category:
                result["error"] = [f"Valid syntax. Successfully decode AST when it should not."]
                result["error_type"] = "irrelevance_error:decoder_success"
            else:
                result["error"] = [f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"]
                result["error_type"] = "relevance_error:decoder_failed"
        return result

    def executable_check(self, action) -> dict:
        decode_error = None
        try:
            decoded_action = self.handler.decode_execute(action)
        except Exception as e:
            decode_error = str(e)
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": False,
                "error": [f"Failed to decode executable. {decode_error}"],
                "error_type": "executable_decoder:decoder_failed",
                "prompt": self.entry,
                "action_raw": str(action),
            }

        if is_rest(self.category):
            if not is_rest_format_output(decoded_action):
                return {
                    "id": self.id,
                    "model_name": self.model_name,
                    "category": self.category,
                    "valid": False,
                    "error": [
                        "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                    ],
                    "error_type": "executable_decoder:rest_wrong_output_format",
                    "prompt": self.entry,
                    "action_raw": str(action),
                    "action_decoded": str(decoded_action),
                }
            checker_result = executable_checker_rest(decoded_action[0], idx)
        else:
            if not is_executable_format_output(decoded_action):
                return {
                    "id": self.id,
                    "model_name": self.model_name,
                    "category": self.category,
                    "valid": False,
                    "error": [
                        "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                    ],
                    "error_type": "executable_decoder:wrong_output_format",
                    "prompt": self.entry,
                    "action_raw": str(action),
                    "action_decoded": str(decoded_action),
                }
            checker_result = executable_checker_non_rest(
                decoded_action, self.entry, self.category
            )

        return {
            "id": self.id,
            "model_name": self.model_name,
            "category": self.category,
            "valid": checker_result["valid"],
            "error": checker_result["error"],
            "error_type": checker_result["error_type"],
            "prompt": self.entry,
            "action_raw": str(action),
            "action_decoded": str(decoded_action),
            "model_executed_output": (
                checker_result["model_executed_output"]
                if "model_executed_output" in checker_result
                else None
            ),
        }

    def multi_turn_check(self, action) -> dict:
        decode_error = None
        tmp_ground_truth = self.ground_truth[:self.task_progress]
        action_list_decoded = []

        try:
            action_list_decoded = self.handler.decode_execute(action)
            self.func_calls_in_turns[self.task_progress - 1].append(action_list_decoded)
            if is_empty_execute_response(action_list_decoded) and not is_empty_execute_response(self.ground_truth[self.task_progress - 1]):
                self.history['message'].extend([{"role": "system", "content": f"Some more functions should be called to solve the user's query."}])
        except Exception as e:
            decode_error = str(e)
            self.history["message"].extend([{"role": "system", "content": "Assistant's response cannot be decoded: " + decode_error}])
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": False,
                "error": [f"Failed to decode executable. {decode_error}"],
                "error_type": "multi_turn_decoder:decoder_failed",
                "prompt": self.entry,
                "action_raw": str(action),
            }

        multi_turn_result = multi_turn_checker(
            self.func_calls_in_turns[:self.task_progress], 
            self.ground_truth[:self.task_progress], 
            self.entry, 
            self.category, 
            self.model_name
            )
        if not multi_turn_result["valid"]: print(f"error: {multi_turn_result['error']}")

        if contain_multi_turn_irrelevance(self.category):
            irrelevance_result = multi_turn_irrelevance_checker(action_list_decoded, tmp_ground_truth, self.task_progress)
        else:
            irrelevance_result = {"valid": True}

        if not irrelevance_result["valid"] and multi_turn_result["valid"]:
            valid = False
            error = [irrelevance_result["error"]]
            error_type = irrelevance_result["error_type"]
        elif irrelevance_result["valid"] and not multi_turn_result["valid"]:
            valid = False
            error = [multi_turn_result["error"]]
            error_type = multi_turn_result["error_type"]
        elif not irrelevance_result["valid"] and not multi_turn_result["valid"]:
            valid = False
            error = [irrelevance_result["error"] + "\n" + multi_turn_result["error"]]
            error_type = irrelevance_result["error_type"] + "\n" + multi_turn_result["error_type"]
        else:
            valid = True
            error = []
            error_type = ""

        self.history["message"].append({"role": "tool", "content": multi_turn_result["latest_step_execution_result"]})
        if valid:
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": valid,
                "prompt": self.entry,
                "action_raw": str(action),
                "action_decoded": str(action_list_decoded),
                "ground_truth": tmp_ground_truth,
                "execution_results": multi_turn_result["execution_results"],
                "latest_step_execution_result": multi_turn_result["latest_step_execution_result"],
            }
        else:
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": valid,
                "error": error,
                "error_type": error_type,
                "prompt": self.entry,
                "action_raw": str(action),
                "action_decoded": str(action_list_decoded),
                "ground_truth": tmp_ground_truth,
                "execution_results": multi_turn_result["execution_results"],
                "latest_step_execution_result": multi_turn_result["latest_step_execution_result"],
            }

    def ast_check(self, action) -> dict:
        decode_error = None
        try:
            decoded_action = self.handler.decode_ast(action, self.language)
        except Exception as e:
            decode_error = str(e)
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": False,
                "error": [f"Invalid syntax. Failed to decode AST. {decode_error}"],
                "error_type": "ast_decoder:decoder_failed",
                "prompt": self.entry,
                "action_raw": str(action),
            }

        decoded_output_valid = is_function_calling_format_output(decoded_action)
        if not decoded_output_valid:
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": False,
                "error": [
                    "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                ],
                "error_type": "ast_decoder:decoder_wrong_output_format",
                "prompt": self.entry,
                "action_raw": str(action),
                "action_decoded": str(decoded_action),
            }
        else:
            checker_result = ast_checker(
                self.function,
                decoded_action,
                self.ground_truth,
                self.language,
                self.category,
                self.model_name,
            )
            return {
                "id": self.id,
                "model_name": self.model_name,
                "test_category": self.category,
                "valid": checker_result["valid"],
                "error": (checker_result["error"] if not checker_result["valid"] else None),
                "error_type": checker_result["error_type"] if not checker_result["valid"] else None,
                "prompt": self.entry,
                "action_raw": str(action),
                "action_decoded": str(decoded_action),
                "ground_truth": self.ground_truth,
            }

    def transition_and_reward(self, results: list[dict] | dict):
        obs = ""
        bug_free = True
        if isinstance(results, list) and len(results) == 1:
            result = results[0]
        else:
            result = results
        if not result["valid"]:
            bug_free = False

        if is_multi_turn(self.category):
            if bug_free:
                # update task progress to next turn
                self.task_progress += 1
                if self.task_progress > len(self.ground_truth):
                    # if no more questions
                    dones = np.ones((self.n_agents), dtype=bool)
                    obs = "Well done! All the questions have been correctly answered."
                else:
                    # if there are more questions
                    dones = np.zeros((self.n_agents), dtype=bool)
                    if str(self.task_progress) in self.holdout_function:
                        obs = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=self.holdout_function[str(self.task_progress)])
                    else:
                        obs = self.question[self.task_progress - 1][0]["content"]
                rewards = [0 if idx != self.n_agents - 1 else 1 for idx in range(self.n_agents)]
                self.history["message"].extend([{"role": "user", "content": obs}])
                self.steps_in_turn = 1 # reset steps for next turn
            else:
                if self.steps_in_turn < self.max_steps:
                    # error but still within self.max_steps
                    dones = np.zeros((self.n_agents), dtype=bool)
                    obs = result["error"][0]
                    rewards = [0 for _ in range(self.n_agents)]
                    self.steps_in_turn += 1
                else:
                    # force reset if cannot complete the query of this turn within self.max_steps
                    dones = np.ones((self.n_agents), dtype=bool)
                    obs = result["error"][0]
                    rewards = [0 if idx != self.n_agents - 1 else -1 for idx in range(self.n_agents)]
                    self.steps_in_turn = 1

        obs = self.handler.format_prompt(self.history['message'], self.history['function'])
        next_obs = np.array([obs for _ in range(self.n_agents)], dtype=np.object_)
        # if (dones == np.ones((self.n_agents), dtype=bool)).any():
        #     # print("$" * 50)
        #     # pprint.pp(self.history['message'])
        #     print("$" * 50)
        #     print(f"functions in steps: {self.func_calls_in_turns[:self.task_progress]}")
        #     print(f"ground truth in steps: {self.ground_truth[:self.task_progress]}")
        #     print(f"error type: {result['error_type']}")
        #     print(f"error: {result['error']}")
        return next_obs, rewards, dones