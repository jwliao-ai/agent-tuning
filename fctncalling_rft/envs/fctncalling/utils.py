from enum import Enum, unique
import re
import abc
import time
import hashlib
from fctncalling_rft.envs.fctncalling.constant import *
from pathlib import Path
from typing import Any, List, Dict, Callable, TypedDict, Union

from pydantic_core import CoreSchema, core_schema
from pydantic import GetCoreSchemaHandler, BaseModel

################### Status classes ###################


@unique
class TaskStatus(Enum):
    UNDO = 0  # not executed
    DOING = 1  # executing
    SUCCESS = 2  # success
    FAIDED = 3  # failed

    def is_undo(self) -> bool:
        return self == TaskStatus.UNDO

    def is_doing(self) -> bool:
        return self == TaskStatus.DOING

    def is_success(self) -> bool:
        return self == TaskStatus.SUCCESS

    def is_failed(self) -> bool:
        return self == TaskStatus.FAIDED


@unique
class MemStage(Enum):
    INPUT = "input"  # user input
    PLAN = "plan"  # task plan
    SUB_TASK = "sub_task"  # current sub task
    SELECT = "select"  # user select
    EXECUTE = "execute"  # tool execute
    ASKING = "asking"  # interaction with user

    def __str__(self) -> str:
        return self.value


@unique
class AgentStatus(Enum):
    WAITTINGTASK = "WAITTINGTASK"
    RUNNING = "RUNNING"
    WAITTINGOBSERVATION = "WAITTINGOBSERVATION"
    WAITTINGAPP = "WAITTINGAPP"
    WAITTINGPARAM = "WAITTINGPARAM"

    @property
    def is_waiting_task(self) -> bool:
        return self.value.startswith("WAITTIN")

    def __str__(self) -> str:
        return self.value


@unique
class InstructionCategory(Enum):
    TOOLUSE = "TOOLUSE"
    PARAM_COMPLETE = "PARAM_COMPLETE"
    OTHER = "OTHER"


@unique
class ActionStatus(Enum):
    ASKING = 0  # missing param
    OK = 1  # param ok
    OTHER = 2  # other error


################### Struct classes ###################


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __deepcopy__ = None

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(dict))


class BaseData(BaseModel):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                kwargs[key] = DotDict(value)
        super().__init__(*args, **kwargs)


class Observation(BaseData):
    status: str
    message: DotDict = {}

    @property
    def is_success(self) -> bool:
        return self.status.lower() == "success"

    def create_with(self, **kwargs):
        kwargs["status"] = kwargs.get("status", self.status)
        kwargs["message"] = kwargs.get("message", DotDict(self.message))
        return Observation(**kwargs)


class ToolAction(BaseData):
    name: str
    intent: str = None
    app: str = None
    plugin_name: str = None
    parameters: DotDict = {}
    meta_data: DotDict = {}

    def __str__(self):
        param_text = []
        for k, v in self.parameters.items():
            if isinstance(v, str):
                param_text.append(f'{k}="{v}"')
            else:
                param_text.append(f"{k}={v}")
        param_text = ", ".join(param_text)
        return f"{self.name}({param_text})"


class TaskIntent(BaseData):
    name: str
    app: str = None


class TaskNode(BaseModel):
    content: str
    extra_content: str = ""
    intent: TaskIntent = None
    status: TaskStatus = TaskStatus.UNDO
    father: "TaskNode" = None
    children: List = []
    executable: bool = False

    tool_candidate: List = []
    tool_action: ToolAction = None
    observation: Observation = None
    action_status: ActionStatus = None
    tool_response: str = None

    def add_children(self, task_node: "TaskNode"):
        self.children.append(task_node)
        task_node.father = self

    @classmethod
    def from_dict(cls, item: Dict):
        task = item["task"]
        task_node = TaskNode(
            content=task,
            status=TaskStatus.UNDO,
            executable=item.get("executable", False),
        )
        for child in item.get("sub_task", []):
            child_node = TaskNode.from_dict(child)
            task_node.add_children(child_node)
        return task_node

    def to_dict(self):
        item = dict(
            task=self.content,
            status=self.status.name,
            executable=self.executable,
            sub_task=[],
        )
        for child in self.children:
            item["sub_task"].append(child.to_dict())
        return item

    def get_depth(self):
        if self.father == None:
            return 0
        return self.father.get_depth() + 1

    def get_subtree_size(self):
        if self.children == []:
            return 1
        now_size = 1
        for child in self.children:
            now_size += child.get_subtree_size()
        return now_size

    def get_task(self, status: TaskStatus, leaf_first: bool = True):
        if leaf_first:
            if self.children:
                for child in self.children:
                    task = child.get_task(status=status, leaf_first=leaf_first)
                    if task is not None:
                        return task
            else:
                if self.status == status:
                    return self
        else:
            if self.status == status:
                return self
            else:
                for child in self.children:
                    task = child.get_task(status=status, leaf_first=leaf_first)
                    if task is not None:
                        return task

    def __str__(self):
        child_str = ", ".join([str(child) for child in self.children])
        if self.father is None:
            father_str = None
        else:
            father_str = self.father.content
        return f"TaskNode(content={self.content}, status={self.status.name}, executable={self.executable}, father={father_str}, children=[{child_str}])"

    def __repr__(self):
        return self.__str__()


class TaskTree:
    def __init__(self, root=None):
        root = root or TaskNode(content="")
        self.root: TaskNode = root

    @classmethod
    def from_list(cls, sub_tasks: List):
        tasks = [TaskNode(content=sub_task, executable=True) for sub_task in sub_tasks]
        if len(tasks) > 1:
            for i in range(len(tasks) - 2, -1, -1):
                tasks[i].children.append(tasks[i + 1])
        return TaskTree(root=tasks[0])

    @classmethod
    def from_dict(cls, item: Dict):
        task_node = TaskNode.from_dict(item)
        return TaskTree(root=task_node)

    @staticmethod
    def _get_node(node: TaskNode, idx: int):
        if idx == 0:
            return node
        for node in node.children:
            idx_node = TaskTree._get_node(node, idx - 1)
            if idx_node is not None:
                return idx_node
        return None

    def get_node(self, idx: int):
        """获取指定idx的子节点"""
        if idx < 0:
            return None
        return TaskTree._get_node(self.root, idx)

    @staticmethod
    def _to_list(node: TaskNode, task_list: List, only_content: bool = True):
        if only_content:
            task_list.append(node.content)
        else:
            task_list.append(node)
        for node in node.children:
            TaskTree._to_list(node, task_list, only_content=only_content)

    def to_list(self):
        task_list = []
        TaskTree._to_list(self.root, task_list)
        return task_list

    def node_list(self):
        task_list = []
        TaskTree._to_list(self.root, task_list, only_content=False)
        return task_list

    def to_dict(self):
        return self.root.to_dict()

    def get_undo_task(self, leaf_first: bool = True):
        return self.root.get_task(status=TaskStatus.UNDO, leaf_first=leaf_first)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(root=TaskNode(content='{self.root.content}'))"
        )


# class AgentData(TypedDict):
#     uid: str  # 用户ID
#     sid: int  # 会话ID
#     instruction: str  # 任务指令
#     plan: TaskTree  # 规划的子任务树
#     doing_task: TaskNode  # 正在处理的子任务

#     executor: Callable  # 工具执行器
#     msg_callback: Callable  # 显示回调函数
