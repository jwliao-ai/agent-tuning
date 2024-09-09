# Function Calling Environment for RL Fine-tuning

## Introduction

This repository implements the function calling environment for RL fine-tuning through reinforcement learning, specifically APPO and TPPO.

<!-- **features**: This environment has high scalability. -->

## Installation

Create a virtual environment with `conda create -n your_env_name python=3.10.13`. Then run `pip install -r requirements.txt` to install the dependencies.

*Note: I did use python 3.10.13 for this project, and for the package versions, you can adapt it to your needs.*

## Instructions

### Dataset

The dataset should be processed and constructed by yourself. And it can be based on various of datasets like xlam-function-calling-60k, ToolBench, etc.

The structure of the dataset should be a list of dicts like this:
```
{
    "query": the user query,
    "answers": {
        "tool_list": a list of answer tools.
        "param_filling": a list of parameter completions.
    }
}
```
for example:
```
{
    "query": "Where can I find live giveaways for beta access and games?",
    "answers": {
        "tool_list": [
            "live_giveaways_by_type",
            "live_giveaways_by_type"
        ],
        "param_filling": [
            {
                "name": "live_giveaways_by_type",
                "arguments": {
                    "type": "beta"
                }
            },
            {
                "name": "live_giveaways_by_type",
                "arguments": {
                    "type": "game"
                }
            }
        ]
    }
}
```

### Tool Inventory

The tool inventory should also be processed and constructed by yourself. And it can be based on various of datasets like xlam-function-calling-60k, ToolBench, etc.

The structure of the tool inventory should be a list of dicts like this:
```
{
    "name": the name of the tool,
    "description": the description of the tool,
    "required_param": a list of the required parameters of the tool,
    "optional_param": a list of the optional parameters of the tool,
}
```
for example:
```
{
    "name": "search_torrents",
    "description": "Search for torrents based on given keywords using the RapidAPI service.",
    "required_param": [
        {
            "keywords": {
                "description": "Keywords to search for torrents.",
                "type": "str",
                "default": "Meg 2 The Trench"
            }
        },
        {
            "quantity": {
                "description": "Number of torrent results to return. Maximum value is 40.",
                "type": "int",
                "default": "40"
            }
        }
    ],
    "optional_param": [
        {
            "page": {
                "description": "Page number for paginated results. Defaults to 1.",
                "type": "int, optional",
                "default": "1"
            }
        }
    ]
},
```

*PS: if the dataset you are processing has more info than the required info above, you can add more info to the dataset and tool inventory for any other training purpose if you want for sure.*

## Train

Run the following command to train the model on your dataset and tool inventory and log the results:

```bash
TOKENIZERS_PARALLELISM=true 
python -u train_fctncalling.py \
        --dataset_name your_dataset_name \ # required
        --dataset_path your_dataset_path \ # required
        --tool_inventory_path your_tool_inventory_path \ # required
        --model_name_or_path your_model_name_or_path \ # required
        --embedding_model your_embedding_model_name_or_path \ # required
        --algorithm_name "APPO" \ # or "TPPO"
        --ppo_epoch 1 \
        --num_mini_batch 4 \
        --experiment_name your_experiment_name \ 
        --seed your_seed \
```

The results (learning curves) will be saved in `./results/your_experiment_name/fctncalling_env/your_dataset_name/algorithm_name` folder. And you can use tensorboard to visualize the results.

## To be updated

- Implementation of more **evaluation metrics**, e.g. AST, EXEC, RELEVANCE, etc.
- **Reward normalization**.
- Implementation of different agents for different family models like Deepseek, Qwen, etc.
- Support for **questioning** stage.
- Support for **multi-turn** conversation learning.
- Support for multi-adapter switching. (centralized multi-agent learning).
- Support for multi-GPU training.
