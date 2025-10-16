"""
This script implements the self-editing process for the SEAL project.

The script is divided into two main phases:

Phase 1: Generate configs using a self-edit model.
In this phase, the script uses a language model to generate a set of "self-edit"
configurations for each task. These configurations specify how to augment the
training data and what training parameters to use.

Phase 2: Train models using the generated configs.
In this phase, the script uses the generated configurations to train a set of
LoRA models. Each model is trained on a different augmented version of the
training data with different training parameters.

The script is designed to be run from the command line and takes several
arguments, including the name of the experiment, the path to the challenge and
solution files, the name of the model to use, and the number of tasks and
self-edits to perform.
"""
import os
import re
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from builtins import input
import argparse

from peft import LoraConfig
import arclib
from arclib.arc import Example, Task
from arclib.arc import (
    make_submission,
    read_tasks_from_single_file,
    to_list,
    to_tuple,
)
from arclib.representers import (
    CompositeRepresenter,
    ConnectedComponentRepresenter,
    DelimitedGridRepresenter,
    DiffExampleRepresenter,
    GridRepresenter,
    ImageTaskRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextTaskRepresenter,
    TextExampleRepresenter,
    WordGridRepresenter,
)
from arclib.messagers import GPTTextMessageRepresenterForBarc, GPTTextMessageRepresenterV2
from arclib.update_model import TTT
from inference.preprocess import get_preprocessed_tasks_single

from arclib.voting import vote
from arclib.eval import evaluate
from inference.engine_vllm import get_sampling_params, initialize_engine, process_requests
from inference.preprocess import get_preprocessed_tasks

import itertools
from typing import List

import numpy as np

from arclib.arc import Task
from arclib.augmenters import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from arclib.messagers import MessageRepresenter

from vllm import LLM, SamplingParams

from utils.prompts import self_edit_prompt, system_message


def mode_array(array_list):
    """Return the most common array from a list of arrays."""
    tuple_shape_list = [(tuple(arr.flatten()), arr.shape) for arr in array_list]
    most_common_tuple_shape, _ = Counter(tuple_shape_list).most_common(1)[0]
    most_common_tuple, original_shape = most_common_tuple_shape
    mode_arr = np.array(most_common_tuple).reshape(original_shape)
    return mode_arr


def read_tasks_from_folder(task_folder: str, test: bool = False) -> List[Task]:
    """Read tasks from a folder of JSON files."""
    all_tasks = []
    for file in glob.glob(f"{task_folder}/*.json"):
        basename = os.path.basename(file)
        idx = basename.replace(".json", "")
        tasks = read_tasks_from_file(file, test=test)
        for i, task in enumerate(tasks):
            task.name = idx + "-" + str(i)
        all_tasks += tasks
    return all_tasks


def read_tasks_from_single_file(
    challenge_file: str, test: bool = False, solution_file: Optional[str] = None
) -> List[Task]:
    """Read tasks from a single JSON file with optional solutions."""
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if solution_file is not None:
        test = False
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks = []
    for task_name, subtasks in data.items():
        parsed_tasks = Task.read_tasks_from_dict(subtasks, test=test)
        for i, task in enumerate(parsed_tasks):
            task.name = task_name + "-" + str(i)
            all_tasks.append(task)

    return all_tasks


def read_tasks_from_file(task_file: str, test: bool = False) -> List[Task]:
    """Read tasks from a JSON file."""
    with open(task_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Task.read_tasks_from_dict(data, test=test)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    #print("Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply))
    return augmenters_to_apply

def _tokenize_and_process(text: str, tokenizer):
    """
    Tokenize text and set up labels for instruction fine-tuning.
    Specifically looks for assistant header token sequence to identify the response part.
    """
    # Tokenize the entire string
    outputs = tokenizer(
        text,
        truncation=True,
    )
    input_ids = outputs["input_ids"]
    
    # Find the special sequence: token 128007 followed immediately by 271
    # This is "<|start_header_id|>assistant<|end_header_id|>"
    special_indexes = []
    for i in range(len(input_ids) - 1):
        if input_ids[i] == 128007 and input_ids[i + 1] == 271:
            special_indexes.append(i + 1)  # include the 271 token in the conditioning
            
    
    special_index = special_indexes[2] if len(special_indexes) == 4 else None
    # If special sequence not found, handle the error
    if special_index is None:
        print("Warning: Special sequence not found, using fallback strategy")
        # Fallback: set labels for the last 20% of tokens
        assert False
    
    # Create labels: we want the model to predict tokens after the special sequence
    labels = input_ids.copy()
    for j in range(special_index + 1):
        labels[j] = -100
    
    outputs["labels"] = labels
    return outputs

def format_and_filter(formatter, tokenizer, task, train_on_input: False):
    task = formatter.encode(task)
    data = {"input": task[0], "output": task[1]}
    task_text = tokenizer.apply_chat_template(task[0] + [task[1]], tokenize=False, add_generation_prompt=True)
    #messages = arc_to_messages(data, train_on_input=False)
    outputs = _tokenize_and_process(task_text, tokenizer)
    data["total_tokens"] = len(outputs["input_ids"])
    data["full_text"] = task_text
    return data


def get_test_time_train_data(
    original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0
) -> List[Task]:
    """
    Generates a set of training tasks for a given original task.

    This function creates a set of training tasks by applying a series of
    augmentations to the original task. The augmentations include basic
    transformations like rotations and flips, as well as more complex
    augmentations like increasing the resolution of the images.

    The function also creates new training tasks by leaving out one or more
    of the original training examples. This is a form of leave-one-out
    cross-validation and helps to improve the generalization of the model.
    """
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)

    # Create new tasks by leaving out n-1 training examples.
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]
        for comb in combs:
            initial_tasks.append(
                Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[i])
            )

    # Apply augmentations to the initial tasks.
    augmented_tasks = []
    for augmenter in augmenters:
        for task in initial_tasks:
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                continue
            augmented_tasks.append(task)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))

    # Permute the colors and examples of the augmented tasks.
    color_and_permute_augmented_tasks = []
    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)

    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks

    augmented_tasks = list(set(augmented_tasks))

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
    max_tokens: int = 8192,
):

    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    formatted_data = []
    for task in train_data:
        formatted = format_and_filter(formatter, tokenizer, task, train_on_input=False)
        if formatted["total_tokens"] < max_tokens:
            formatted_data.append(formatted)

    return formatted_data


def process_task(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: List[int],
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)
    
    train = []
    # Generate training data for each n in leave_n
    for n in leave_n:
        leave_n_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer, leave_n=n, permute_n=permute_n, seed=seed
        )
        train.extend(leave_n_train_data)

    # Shuffle and limit the total number of examples if needed
    if len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]

    return train

def get_prompt(task: Task, system_message: str, self_edit_prompt: str):
    train_examples = task.serialize()['train']
    formatted_examples = ""

    for example in train_examples:
        # Format input grid
        input_grid = example['input']
        input_str = "Input:\n"
        for row in input_grid:
            input_str += " ".join(map(str, row)) + "\n"
        
        # Format output grid
        output_grid = example['output']
        output_str = "\nOutput:\n"
        for row in output_grid:
            output_str += " ".join(map(str, row)) + "\n"
        
        # Combine with separator
        formatted_examples += input_str + output_str + "\n"

    user_message = formatted_examples
    user_message = user_message + "------\n\n" + self_edit_prompt
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def main(experiment_name, skip_repeated_configs, challenge_file, solution_file, model_name, n_tasks, n_self_edits_per_task):
    # lora config
    lora_config = LoraConfig(
        r=128, 
        lora_alpha=16,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
    )

    # training config
    batch_size = 2
    gradient_accumulation_steps = 1
    lr_scheduler_type = "cosine"

    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )

    representer = GPTTextMessageRepresenterV2(task_representer=standard_formatter)

    # Load tasks
    tasks = read_tasks_from_single_file(
        challenge_file=challenge_file, 
        solution_file=solution_file
    )

    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Phase 1: Generate configs using self-edit model
    print("Phase 1: Generating configs using self-edit model...")
    self_edit_model = LLM(model=model_name)
    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.8,
    )

    # Dictionary to store explored configs per task
    explored_configs = {}
    task_configs = {}  # Store full configs for each task

    for i in range(n_tasks):
        task = tasks[i]
        prompt = get_prompt(task, system_message, self_edit_prompt)
        
        # Get the base task name (without -0 or -1 suffix) skip if it has -1 suffix
        base_task_name = task.name
        if base_task_name.endswith("-0"):
            base_task_name = base_task_name[:-2]
        if base_task_name.endswith("-1"):
            continue
            
        # Initialize config tracking for this task
        if base_task_name not in explored_configs:
            explored_configs[base_task_name] = set()
            task_configs[base_task_name] = []
        
        while len(task_configs[base_task_name]) < n_self_edits_per_task:
            response = self_edit_model.generate(prompt, sampling_params=sampling_params)

            try:
                config = json.loads(response[0].outputs[0].text)
            except json.JSONDecodeError as e:
                continue

            # Convert config to a hashable format (tuple of tuples) for storing in set
            config_key = (
                ("data_generation", tuple(sorted(config["data_generation"].items()))),
                ("training", tuple(sorted(config["training"].items())))
            )
            
            # Skip if this config was already explored for this task
            if skip_repeated_configs and config_key in explored_configs[base_task_name]:
                print(f"Skipping already explored config for task {base_task_name}")
                continue
                
            # Add config to explored set and store full config
            explored_configs[base_task_name].add(config_key)
            task_configs[base_task_name].append({
                "config": config,
                "prompt": prompt,
                "response": response[0].outputs[0].text,
                "token_ids": response[0].outputs[0].token_ids
            })
            print(f"New config for task {base_task_name}:", config)

    # Delete self-edit model to free memory
    del self_edit_model
    print("Phase 1 complete.")

    # Phase 2: Train models using generated configs
    print("\nPhase 2: Training models using generated configs...")
    
    # setup ttt 
    ttt = TTT(
        model_name=model_name,
        lora_config=lora_config
    )

    final_configs_and_indices = {}
    # Train models for each task using its configs
    for base_task_name, configs in task_configs.items():
        task = next(t for t in tasks if t.name.startswith(base_task_name))
        task_ttt = 0
        curr_task_configs = {}
        for config_data in configs:
            config = config_data["config"]
            try:
                augmenters_to_apply = get_augmenters(
                    include_basic=config["data_generation"]["use_basic_augmentations"],
                    include_size=config["data_generation"]["use_size_augmentations"],
                    include_chain=config["data_generation"]["use_chain_augmentations"],
                    include_repeat=config["data_generation"]["use_repeat_augmentations"]
                )
            except Exception as e:
                print(f"Error getting augmenters for task {base_task_name}: {e}")
                augmenters_to_apply = get_augmenters(
                    include_basic=False,
                    include_size=False,
                    include_chain=False,
                    include_repeat=False
                )
                config["training"]["num_train_epochs"] = 0

            train_data = process_task(
                task=task,
                augmenters=augmenters_to_apply,
                formatter=representer,
                tokenizer=tokenizer,
                leave_n=[1,2],
                permute_n=1,
                Nmax=250,
                seed=0
            )

            if len(train_data) == 0:
                continue

            task_text_list = [data["full_text"] for data in train_data]

            # check if the keys "strategy" and "num_train_epochs" , "learning_rate" are in the config
            if "strategy" not in config["training"] or "num_train_epochs" not in config["training"] or "learning_rate" not in config["training"]:
                print(f"Skipping training for task {base_task_name} because the training config is not valid")
                config["training"]["num_train_epochs"] = 0
                config["training"]["learning_rate"] = 0
                config["training"]["strategy"] = "train_using_all_tokens"

            if config["training"]["strategy"] not in ["train_using_all_tokens", "train_using_output_tokens"]:
                print(f"Skipping training for task {base_task_name} because the training strategy is not valid")
                config["training"]["num_train_epochs"] = 0

            # if the number of steps is greater than 250, then we create a dummy lora  
            if config["training"]["num_train_epochs"] * len(train_data) // 2 > 375:
                print(f"Skipping training for task {base_task_name} because the number of steps is greater than 375")
                config["training"]["num_train_epochs"] = 0

            adapter_path = ttt.update_model(
                task_text_list=task_text_list,
                output_dir=f"loras/self-edit/{experiment_name}/{base_task_name}/{task_ttt}",
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=config["training"]["learning_rate"],
                num_train_epochs=config["training"]["num_train_epochs"],
                lr_scheduler_type=lr_scheduler_type,
                loss_on_all_tokens=config["training"]["strategy"] == "train_using_all_tokens"
            )

            curr_task_configs[task_ttt] = config_data
            task_ttt += 1

        final_configs_and_indices[base_task_name] = curr_task_configs
    # Delete ttt to free memory
    del ttt

    # Save final configs and indices to file
    configs_file = os.path.join(f"loras/self-edit/{experiment_name}", "final_configs_and_indices.json")
    os.makedirs(os.path.dirname(configs_file), exist_ok=True)
    with open(configs_file, "w") as f:
        json.dump(final_configs_and_indices, f)
    
    print("Training complete. Final configs and indices saved to:", configs_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run self-edit training with specified parameters')
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--skip_repeated_configs', action='store_true',
                      help='Whether to skip repeated configs')
    parser.add_argument('--challenge_file', type=str, required=True,
                      help='Path to the challenge file')
    parser.add_argument('--solution_file', type=str, required=True,
                      help='Path to the solution file')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use')
    parser.add_argument('--n_tasks', type=int, required=True,
                      help='Number of tasks to process')
    parser.add_argument('--n_self_edits_per_task', type=int, required=True,
                      help='Number of self-edits per task')

    args = parser.parse_args()
    
    main(
        experiment_name=args.experiment_name,
        skip_repeated_configs=args.skip_repeated_configs,
        challenge_file=args.challenge_file,
        solution_file=args.solution_file,
        model_name=args.model_name,
        n_tasks=args.n_tasks,
        n_self_edits_per_task=args.n_self_edits_per_task
    )
    
   
 
    
