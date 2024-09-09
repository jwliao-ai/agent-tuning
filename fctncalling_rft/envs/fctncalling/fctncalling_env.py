import sys
import json

# sys.path.append("../../../")
import numpy as np
import random
import re
import pprint
from json_repair import repair_json
from mat.envs.fctncalling.prompts import *
from mat.envs.fctncalling.utils import *
from mat.envs.fctncalling.retriever import Retriever
from copy import deepcopy


class FctnCallingEnv:

    def __init__(self, flag, rank, dataset_path, retriever: Retriever, no_nan=True):
        self.rank = rank
        self.no_nan = no_nan
        self.n_agents = 1
        self.max_step = 8
        self.retriever = retriever
        self.complex_query = False
        self.multi_turn = False
        self.multi_func = False
        self.retrieval_top_k = 6
        self.dataset = []
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        self.stage = 0
        self.step_count = 0
        self.instruction = ""
        self.history = []
        self.final_status = ""
        self.available_tools_text = ""

        # self.log_file = f"../../mat/envs/fctncalling/cache/{flag}_{rank}.py"
        # with open(self.log_file, "w") as f:
        #     f.write("")

    def reset(self):
        self.retrieved_funcs = []
        self.retrieved_funcs_text = ""
        self.tool_select_task = ""
        self.tool_schema_mapping = {}
        self.task_done = False
        self.traj = random.choice(self.dataset)
        self.task = self.traj["query"]
        self.answers = self.traj["answers"]
        self.history = [{"query": self.task}]
        if (
            self.complex_query == False
        ):  # next stage: tool selection (for easy query, skip task decomposition)
            self.plans = TaskTree.from_list([self.task])
            self.cur_task = self.plans.get_undo_task(leaf_first=True)
            self.sub_task = self.handle_placeholder(self.cur_task)
            self.retrieved_funcs = self.retriever.get_relevant_documents(
                self.sub_task, self.retrieval_top_k
            )
            if not set(self.answers["tool_list"]).issubset(
                set([rtool["name"] for rtool in self.retrieved_funcs])
            ):
                self.reset()
            for tool in self.retrieved_funcs:
                self.retrieved_funcs_text += TOOL_LIST_SCHEMA.format(
                    name=tool["name"], description=tool["description"]
                )
                self.tool_schema_mapping[tool["name"]] = tool
            # pprint.pp(self.answers['tool_list'])
            obs1 = SELECT_TOOL_EN.format(
                query=self.sub_task + self.cur_task.extra_content,
                tool_list=self.retrieved_funcs_text.strip(),
            )
            self.stage = 2
        else:  # next stage: task decomposition
            self.available_tools_text = self.available_tools_text.strip()
            obs1 = RAG_LORA_USER_PROMPT_EN.format(
                available_tools=self.available_tools_text, query=self.task
            )

        obs = np.array([obs1], dtype=np.object_)
        self.history = [{"query": self.task}, {"observation": obs1}]
        self.step_count = 0

        return obs

    def step(self, action):
        self.step_count += 1
        bug_free = True
        action = action[0]
        self.history.append({"action": action})

        if (
            self.stage == 0
        ):  # task decomposition done, return observation for intent classification
            # task_list = action.split('<|action|>')[-1].split('<Functions>')[0].strip().split('|task_id|')
            self.plans = TaskTree.from_list(self.task_list)

            if self.cur_task:
                self.cur_task.status = TaskStatus.DOING
                self.cur_task.tool_action = ToolAction(name="")
                task_desc = self.sub_task + self.cur_task.extra_content
            next_obs1 = INTENT_FIX_SHOT_USER_PROMPT_EN.format(
                query=task_desc, intent_list=self.intent_categories
            )
            score = 0
            self.stage = 2

        elif (
            self.stage == 2
        ):  # tool selection done, check if the tool is correct, if not, reselect, if yes, return observation for parameter filling
            self.action_func_names = [
                x.strip() for x in action.strip().split(",") if x.strip() != ""
            ]
            self.action_func_index = 0
            if set(self.action_func_names) == set(self.answers["tool_list"]):
                self.multi_func = True
                self.action_func_names = list(set(self.action_func_names))
                self.cur_func = self.tool_schema_mapping[
                    self.action_func_names[self.action_func_index]
                ]  # here tool_names is a list WITH DEPENDENCY CONSTRAINT
                self.action_func_index += 1
                self.params = []
                param_list = self.cur_func["required_param"]
                param_list.extend(self.cur_func["optional_param"])
                for item in param_list:
                    p_name = list(item.keys())[0]
                    p_meta = item[p_name]
                    self.params.append(
                        dict(
                            name=p_name,
                            **p_meta,
                            required=p_name in self.cur_func["required_param"],
                        )
                    )
                next_obs1 = FILL_PARAM_EN.format(
                    tool_name=self.cur_func["name"],
                    parameters=json.dumps(self.params),
                    query=self.task,
                )
                score = 1
                self.stage = 3
                print(f"correct tool selection.")
            elif set(self.action_func_names).issubset(set(self.answers["tool_list"])):
                next_obs1 = SELECT_MORE_TOOL_EN.format(
                    query=self.task,
                    selected_tools=", ".join(self.action_func_names),
                    tool_list=self.retrieved_funcs_text.strip(),
                )
                score = -0.5
                bug_free = False
            elif not set(self.action_func_names).issubset(
                set(self.answers["tool_list"])
            ):
                hallucinated_tools = [
                    tool_name
                    for tool_name in self.action_func_names
                    if tool_name not in [tool["name"] for tool in self.retrieved_funcs]
                ]
                wrong_tools = [
                    tool_name
                    for tool_name in self.action_func_names
                    if tool_name in [tool["name"] for tool in self.retrieved_funcs]
                    and tool_name not in self.answers["tool_list"]
                ]
                if hallucinated_tools and wrong_tools:
                    next_obs1 = SELECT_TOOL_WRONG_HALLUCINATED_EN.format(
                        query=self.task,
                        selected_tools=", ".join(self.action_func_names),
                        wrong_tools=", ".join(wrong_tools),
                        hallucinated_tools=", ".join(hallucinated_tools),
                        tool_list=self.retrieved_funcs_text.strip(),
                    )
                elif hallucinated_tools and not wrong_tools:
                    next_obs1 = SELECT_TOOL_HALLUCINATED_EN.format(
                        query=self.task,
                        selected_tools=", ".join(self.action_func_names),
                        hallucinated_tools=", ".join(hallucinated_tools),
                        tool_list=self.retrieved_funcs_text.strip(),
                    )
                elif wrong_tools and not hallucinated_tools:
                    next_obs1 = SELECT_TOOL_WRONG_EN.format(
                        query=self.task,
                        selected_tools=", ".join(self.action_func_names),
                        wrong_tools=", ".join(wrong_tools),
                        tool_list=self.retrieved_funcs_text.strip(),
                    )
                else:
                    NotImplementedError
                score = -1
                bug_free = False
            else:
                NotImplementedError

        elif (
            self.stage == 3
        ):  # parameter filling done, check if the parameter is correct, if not, double check, if yes, return observation for tool execution
            param_label = self.get_answer_arguments(self.cur_func["name"])
            # print(f"param label: ")
            # pprint.pp(param_label)
            param_completions = json.loads(repair_json(action.strip()))
            print(f"param completions: {type(param_completions)}")
            if type(param_completions[0]) == list:
                param_completions = param_completions[0]
            print(f"param completions: {type(param_completions)}")
            pprint.pp(param_completions)
            wrong_params = [x for x in param_completions if x not in param_label]
            missing_params = [y for y in param_label if y not in param_completions]
            if not wrong_params and not missing_params:  # correctly filled
                score = 1
                print(f"correct parameter completions.")
                if self.multi_func and self.action_func_index < len(
                    self.action_func_names
                ):
                    self.cur_func = self.tool_schema_mapping[
                        self.action_func_names[self.action_func_index]
                    ]
                    self.action_func_index += 1
                    self.params = []
                    param_list = self.cur_func["required_param"]
                    param_list.extend(self.cur_func["optional_param"])
                    for item in param_list:
                        p_name = list(item.keys())[0]
                        p_meta = item[p_name]
                        self.params.append(
                            dict(
                                name=p_name,
                                **p_meta,
                                required=p_name in self.cur_func["required_param"],
                            )
                        )
                    next_obs1 = FILL_PARAM_EN.format(
                        tool_name=self.cur_func["name"],
                        parameters=json.dumps(self.params),
                        query=self.task,
                    )
                else:
                    self.task_done = True
                    next_obs1 = (
                        f"tool selection and parameter completions are all correct!"
                    )
            elif missing_params and not wrong_params:
                next_obs1 = FILL_MORE_PARAM_EN.format(
                    query=self.task,
                    tool_name=self.cur_func,
                    last_param_completions=action.strip(),
                    parameters=json.dumps(self.params),
                )
                score = -1
                bug_free = False
            else:
                next_obs1 = FILL_PARAM_WRONG_EN.format(
                    query=self.task,
                    tool_name=self.cur_func,
                    last_param_completions=action.strip(),
                    parameters=json.dumps(self.params),
                )
                score = -1
                bug_free = False

        next_obs = np.array([next_obs1], dtype=np.object_)
        if not bug_free:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            if self.task_done:
                score += 2.0 / self.step_count
                dones = np.ones((self.n_agents), dtype=bool)
            else:
                dones = np.zeros((self.n_agents), dtype=bool)

        rewards = [score for _ in range(self.n_agents)]

        self.history.append({"observation": next_obs1})
        infos = [
            {
                "task_done": self.task_done,
                "history": self.history,
                "traj_length": self.step_count,
            }
            for _ in range(self.n_agents)
        ]
        return next_obs, rewards, dones, infos

    def handle_placeholder(self, task: TaskNode):
        """处理多步任务规划任务结果依赖问题"""
        task_desc = task.content
        places = re.findall(
            re.compile(r"\|placeholder_id\|\(([^(?!.*placeholder_id)]+)\)", re.S),
            task_desc,
        )
        for place in places:
            m = re.match("<\|\$s(\d+)\|>(.*)", place)
            if m:
                idx = int(m.group(1)) - 1
                place_res = m.group(2)
            else:
                idx = -1
                place_res = place[0]
            task_desc = task_desc.replace(f"|placeholder_id|({place})", place_res)
            node_list = self.plans.node_list()
            for i in range(len(node_list)):
                if node_list[i].status.is_doing():
                    node_list = node_list[:i]
                    break
            response = node_list[idx].tool_response
            task_desc += f"，{place_res}：{response}"
        return task_desc

    def get_answer_arguments(self, function_name: str) -> list[dict]:
        arguments = []
        for item in self.answers["param_filling"]:
            if item["name"] == function_name:
                arguments.append(item["arguments"])
        return arguments

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
