"""
This script evaluates the performance of the self-edited models.

The script takes as input a set of tasks and a set of LoRA checkpoints, and it
evaluates the performance of each model on each task. The script can be used to
evaluate the performance of the baseline model, as well as the performance of
the self-edited models.

The script is designed to be run from the command line and takes several
arguments, including the path to the data and solution files, the path to the
pretrained checkpoint, the path to the LoRA checkpoints folder, and the number
of self-edits to evaluate.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
)
from vllm.lora.request import LoRARequest

import arclib.messagers
from arclib.arc import (
    make_submission,
    read_tasks_from_single_file,
    to_list,
    to_tuple,
)
import arclib.augmenters  # noqa: F401 to prevent removal by black
from arclib.eval import evaluate
from arclib.messagers import GPTTextMessageRepresenterV2, GPTTextMessageRepresenterForBarc
from arclib.representers import (
    DiffExampleRepresenter,
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
    WordGridRepresenter,
)
from arclib.voting import vote
from inference.engine import get_sampling_params, initialize_engine, process_requests
from inference.preprocess import get_preprocessed_tasks


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--data_file",
    type=str,
    default="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
    help="Data file path to evaluate",
)
parser.add_argument(
    "--solution_file",
    type=str,
    default="/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    help="Solution file path to evaluate",
)
parser.add_argument(
    "--num_examples",
    type=int,
    default=419,
    help="Number of examples to process for limited evaluation.",
)
parser.add_argument(
    "--pretrained_checkpoint",
    type=str,
    default="checkpoints/pretrained/multi_format_model/",
    help="path to the pretrained checkpoint",
)
parser.add_argument(
    "--lora_checkpoints_folder",
    type=str,
    default=None,
    help="LoRA checkpoints folder, if none then base model is used",
)
parser.add_argument(
    "--quantization", type=str, default=None, help="Qusantization type bitsandbytes or none"
)
parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
parser.add_argument(
    "--n_sample", type=int, default=1, help="Number of samples to generate per input"
)
parser.add_argument(
    "--experiment_folder", type=str, default="experiments/tti/new/", help="submission folder"
)
parser.add_argument(
    "--formatter",
    type=str,
    default="arclib.messagers.GPTTextMessageRepresenterV2",
    help="formatter for the task, better to be same with the one used for training",
)
parser.add_argument(
    "--max_lora_rank",
    type=int,
    default=64,
    help="Max lora rank, should be same with the one used for training",
)
parser.add_argument(
    "--include_n",
    type=int,
    nargs="+",
    default=[1],
    help="Which leave-n tasks to include, it is generally 1 for test time trained model, 0 for base model",
)
parser.add_argument(
    "--permute_n",
    type=int,
    default=2,
    help="Number of permutations to generate for each leave-n task",
)
parser.add_argument(
    "--new_format", action="store_true", help="Whether to use the new format or not"
)

parser.add_argument(
    "--barc_format", action="store_true", default=False, help="Whether to use the new format or not"
)
parser.add_argument(
    "--add_diff_format", action="store_true", help="Whether to use the new format or not"
)

parser.add_argument(
    "--use_all_lora", action="store_true", help="single trained lora"
)

parser.add_argument(
    "--n_self_edits", type=int, required=True, help="Number of self edits to evaluate"
)

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# print args
print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

os.makedirs(args.experiment_folder, exist_ok=True)

tasks = read_tasks_from_single_file(args.data_file, solution_file=args.solution_file, test=True)

id_to_lora_path = {}
# get lora paths and filter tasks if necessary
if args.lora_checkpoints_folder is not None:
    id_to_lora_path = {}
    for lora_path in glob.glob(f"{args.lora_checkpoints_folder}/*/0"): # adapter_model.safetensors
        lora_id = lora_path.split("/")[-2]
        id_to_lora_path[lora_id] = lora_path
        lora_dir = os.path.dirname(lora_path)

if args.num_examples is not None:
    # shuffle
    np.random.seed(args.seed)
    np.random.shuffle(tasks)
    tasks = tasks[: args.num_examples]

formatters = []
if args.new_format:
    messager = GPTTextMessageRepresenterV2(
        task_representer=TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
                io_sep=" -> ",
                input_header="",
                output_header="",
                output_footer="#",
                grid_representer=PythonListGridRepresenter(),
            )
        )
    )
    formatters.append(messager)
elif args.barc_format:
    messages = arclib.messagers.GPTTextMessageRepresenterForBarc(
        task_representer=arclib.representers.TextTaskRepresenter(
            example_representer=arclib.representers.TextExampleRepresenter(
            grid_representer=arclib.representers.WordGridRepresenter(),
            input_header="Input:\n",
            output_header="\nOutput:\n",
            io_sep="\n"
        )))
    formatters.append(messages)
else:
    messager = arclib.messagers.GPTTextMessageRepresenterV2()
    formatters.append(messager)

if args.add_diff_format:
    diff_formatter = TextTaskRepresenter(
        example_representer=DiffExampleRepresenter(
            use_output=False,
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )
    input_diff_formatter = GPTTextMessageRepresenterV2(task_representer=diff_formatter)

    formatters.append(input_diff_formatter)

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_checkpoint)

task_name_to_processed_data = get_preprocessed_tasks(
    tasks,
    tokenizer,
    formatters,
    max_tokens=args.max_tokens,
    id_to_lora_path=id_to_lora_path,
    include_n=args.include_n,
    permute_n=args.permute_n,
)
valid_tasks = [info for key, info in task_name_to_processed_data.items() if info["valid"]]
invalid_tasks = [info for key, info in task_name_to_processed_data.items() if not info["valid"]]

print("Len of valid tasks:", len(valid_tasks))
print("Len of invalid tasks:", len(invalid_tasks))
# for each valid task print the length of queries
for info in valid_tasks:
    print(f"{info['task'].name}: Number of Queries: {len(info['queries'])}")

example_task = valid_tasks[0]
example_task_id = example_task["task"].name.split("-")[0]

print("Example Task Information:")
print(f"Task Name: {example_task['task'].name}")
print(f"Number of Queries: {len(example_task['queries'])}")
print("Example Query:" + example_task["queries"][0]["text"])

# lora_path = f"{args.lora_checkpoints_folder}/{example_task_id}/"
# abstract away
inputs_to_the_engine = []
inputs_to_remember = {}
lora_path_idxs = list(id_to_lora_path.keys())

if len(lora_path_idxs) > 0:
    # load one adapter_config.json
    with open(
        id_to_lora_path[lora_path_idxs[0]] + "/adapter_config.json"
    ) as f:
        lora_adapter_config = json.load(f)
else:
    lora_adapter_config = {}

engine = initialize_engine(
    model=args.pretrained_checkpoint,
    quantization=args.quantization,
    max_lora_rank=lora_adapter_config.get("r", args.max_lora_rank),
    enable_lora=args.lora_checkpoints_folder is not None,
    enforce_eager=False,
    lora_target_modules=lora_adapter_config.get("target_modules", None),
)


final_results = {}
for lora_edit_idx in range(args.n_self_edits):

    inputs_to_the_engine = []
    inputs_to_remember = {}

    for i, info in enumerate(valid_tasks):
        name = info["task"].name
        idx, no = name.split("-")
        if args.lora_checkpoints_folder is not None:
            lora_path = id_to_lora_path[idx]
            lora_path = os.path.dirname(lora_path) + f"/{lora_edit_idx}"
            
            # get the parent folder
            if args.use_all_lora:
                lora_path = os.path.join(os.path.dirname(lora_path), "all/")
            lora_index = lora_path_idxs.index(idx)

            s = lora_index + lora_edit_idx
            unique_id = (s * (s + 1)) // 2 + lora_edit_idx
            lora_request = LoRARequest(idx + no + '-' + str(lora_edit_idx), unique_id + 1, lora_path)
        else:
            lora_request = None
        test_inputs = info["queries"]
        for j, test_input in enumerate(test_inputs):
            input_token_length = len(tokenizer.encode(test_input["text"])) - 1
            sampling_param = get_sampling_params(
                tokenizer,
                input_token_length,
                args.max_tokens,
                temperature=args.temperature,
                n=args.n_sample,
            )
            inputs_to_the_engine.append(
                (test_input["text"], sampling_param, lora_request, name + "-" + str(j))
            )
            inputs_to_remember[name + "-" + str(j)] = test_input


    print(f"Number of input queries to the engine: {len(inputs_to_the_engine)}")

    outputs_by_key = process_requests(engine, inputs_to_the_engine)

    for key in list(outputs_by_key.keys()):
        inverter = inputs_to_remember[key]["inverter"]
        if inverter is not None:
            inverter_fn = eval("arclib.augmenters." + inverter)
        else:
            inverter_fn = np.array

        outputs = outputs_by_key[key]
        outputs_by_key[key] = []
        current_formatter_repr = inputs_to_remember[key]["formatter"]
        input = inputs_to_remember[key]["input"]["content"]
        current_formatter = eval(current_formatter_repr)

        for output in outputs:
            output = output.replace("#", "")
            output = output.replace("  ", " ")
            if "```" in output:
                # get things between ``` and ```
                output = output.split("```")[1]
                output = output.strip()
                input = input.split("Here is the input grid for the test example:\nInput:\n")[-1]
                input = input.split("\n\n\nDirectly provide")[0]
                input = input.strip()

            try:
                decoded = current_formatter.task_representer.example_representer.decode(
                    (input, output)
                )
            except Exception as e:
                print(f"Cannot Decode: {e}")
                print(f"Input: {input}")
                print(f"Output: {output}")
                continue

            try:
                pred = to_tuple(inverter_fn(decoded.output))
            except Exception as e:
                print(f"Error: {e}")
                continue

            if decoded is not None:
                outputs_by_key[key].append(
                    {
                        "output": to_tuple(inverter_fn(decoded.output)),
                        "inverter": inverter,
                        "formatter": current_formatter_repr,
                    }
                )

    outputs_by_key = {key: outputs for key, outputs in outputs_by_key.items() if len(outputs) > 0}

    # save
    all_predictions_file = os.path.join(args.experiment_folder, "all_predictions.json")

    with open(all_predictions_file, "w") as f:
        json.dump(outputs_by_key, f)

    outputs = {}
    for task in tasks:
        name = task.name

        to_vote = [out for key, out in outputs_by_key.items() if name in key]
        to_vote = [out for sublist in to_vote for out in sublist]

        if len(to_vote) == 0:
            outputs[name] = [[[0]], [[0]]]
            continue
        else:
            attempt_1, attempt_2 = vote(to_vote)
            outputs[name] = [to_list(attempt_1), to_list(attempt_2)]

    predictions = [outputs[task.name] for task in tasks]

    submission_file = os.path.join(args.experiment_folder, "submission.json")
    make_submission(tasks, predictions, submission_file, number_of_attempts=2)

    print(f"Submission file is saved to {submission_file}")

    # evaluate
    if args.solution_file is not None:
        task_info = evaluate(args.data_file, args.solution_file, submission_file)
        final_results[lora_edit_idx] = task_info

print(final_results)
# save final results
for k in final_results.keys():
    final_results[k] = final_results[k].to_dict()

with open(os.path.join("final_results.json"), "w") as f:
    json.dump(final_results, f)
