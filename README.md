 # Function Calling Environment for RL Fine-tuning (versatile version feat. BFCL)

## Introduction

This repository implements the function calling environment for RL fine-tuning through reinforcement learning, specifically APPO and TPPO.

<!-- **features**: This environment has high scalability. -->

## Installation

Create a virtual environment with `conda create -n your_env_name python=3.10.13`. Then run `pip install -r requirements.txt` to install the dependencies.

*Note: I did use python 3.10.13 for this project, and for the package versions, you can adapt it to your needs.*

## Instructions

### Dataset

The dataset should have a similar structure as the data BFCL uses with the following structure:
```
{
    'id': 'multiple/multi-turn/parallel_java/python/javascript_num',    # use id to choose language and multi-turn or simple or multiple or parallel
    'question': [
        [
            {
                "role": "user",
                "content": "How can I validate user input in a form field with the ID 'userInputField' after the user has finished typing?"
            }
        ],
        # ... if multi turn
    ],
    'function': [
        {
            'name': 'name',
            'description': 'description',
            'parameters': {
                'type': 'type',
                'properties': {
                    'arg_1': {
                        'type': 'type',
                        'description': 'description'
                    },
                    # ... if more args
                'required': [
                    a list of required arg names
                    ]
                }
            },
            'response': {
                'name': 'name',
                'type': 'type',
                'description': 'description',
                'items': {    # if have
                    'type': 'type'
                }
            },
        },
        # ... if multiple
    ]
    'initial_config':{} # if multi turn, as a environment setting (question: how to let the agent know)
    'invoked_classes': [
        'invoked_classes',
        ......
    ]
    'ground_truth': [
        # for multi-turn, a list of str (executable function call)
        # for single-turn, a list of dict (json formatted function call)
    ]
}
```

*PS: if the dataset you are processing has more info than the required info above, you can add more info to the dataset and tool inventory for any other training purpose if you want for sure.*

### Dataset Loading

The environment loads the dataset easily like:
```
with open(dataset_path, "r") as f:
    self.dataset = json.load(f)
```
which means all the data should be included in single file and strictly follow JSON format. This may not be the most general one but works for now.

## Environment process

### Reset

When `done` is ones, the environment resets. And a new entry will be randomly chosen as a new current task. All the questions, ground truth, entry id, observations, actions, history will be refreshed.

### Step

Given the observation, the model (agent) acts (giving a response whcih means 'Step'). The action (or response) will be decoded for ast or execution by the handler, and will be checked and evaluated by some kind of checker according to its category (contained in the entry id). And environment will return a new observation, reward, done, info. The reward can be used for the agent to do training.

### Buffer preparation

After an episode, the buffer will do some preparation for APPO or TPPO. The preparation inlcludes doing Bellman-backup with Action Decomposition (BAD) which is generally called value update.

### Training

After buffer preparation, the trainer will train the agent. Sample a batch of on-policy data from the buffer and do value function training and policy training.

## Training script

Run the following command to train the model on your dataset and tool inventory and log the results:

```bash
CUDA_VISIBLE_DEVICES=0 python train_fctncalling.py \
        --seed 10 \
        --env_name "fctncalling_env" \
        --algorithm_name "TPPO" \ # TPPO or APPO, now only support TPPO
        --experiment_name "default" \
        --num_mini_batch 2 \
        --ppo_epoch 1 \
        --lr 1e-7 \
        --critic_lr 5e-6 \
        --dataset_path path_to_BFCL_v3_multi_turn_base.json_or else \
        --model_name_or_path path_to_model_weight \
        --n_rollout_threads 2 \ # more for efficiency (even number is better)
        --gradient_cp_steps 4 \ # for CUDA memory efficiency
        --max_new_tokens 128 \
        --save_interval 30 \ # interval of saving models
```

The results (learning curves) will be saved in `./results/your_experiment_name/model_type/your_dataset_name/algorithm_name` folder. And you can use tensorboard to visualize the results.

## Current support

- multi-facet checker (evaluation metrics).
- multi-turn support,
- multi-GPU training

## To be updated

- Agents backboned proprietary models like GPT-4, etc.
- Support for **questioning** stage.
- Support for multi-adapter switching.
- Design process for multi-agent -> algo. improvement.