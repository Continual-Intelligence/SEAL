# SEAL - few-shot

This document provides the commands to reproduce the SEAL few-shot learning experiments on the ARC dataset. The goal of these experiments is to demonstrate how SEAL can adapt to new tasks with only a few examples.

The experiments are divided into several steps, including training the base model, evaluating the trained models, and running a baseline evaluation. By following the steps in this guide, you will be able to reproduce the results from the paper.

This document contains the commands to reproduce the SEAL Few Shot Experiments on ARC. 
Code is adopted from: [Ekin's Repo](https://github.com/ekinakyurek/marc/tree/main)

## Environment Variables

Before running the experiments, you will need to set the following environment variables:

*   `DATA_DIR`: This variable should point to the directory where the ARC dataset is located.
*   `TTI_DIR`: This variable should point to the directory where the training text-to-image data is located.
*   `LORA_DIR`: This variable should point to the directory where the LoRA checkpoints will be saved.

Please make sure to set these variables to the correct paths in your environment before running the commands.

## SEAL RL Iteration 1

### 1. Training on 12 Problems (Iteration 1)

Train the base model on 12 problems from ARC train set:

```bash
python self-edit.py \
    --experiment_name=training_set_iteration_1 \
    --challenge_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --n_tasks=12 \
    --n_self_edits_per_task=15
```

### 2. Evaluation on Iteration 1 LoRAs

Evaluate the trained LoRAs from iteration 1:

```bash
python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/training_set_iteration_1 \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/training_set_iteration_1 \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=11 \
    --n_self_edits=15
```

### 3. RestEM on Iteration 1 (8 Epochs)

Run RestEM training for 8 epochs:

```bash
python BC-self-edit.py \
    --configs_and_indices=${LORA_DIR}/self-edit/training_set_iteration_1/final_configs_and_indices.json \
    --results=${LORA_DIR}/self-edit/training_set_iteration_1/final_results.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --lora_rank=16 \
    --lora_alpha=16 \
    --num_train_epochs=8 \
    --per_device_train_batch_size=5 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5
```

### 4. Create Self-Edits on Eval Set (RL Iteration 1, 8 Epochs)

Generate self-edits on evaluation set using the 8-epoch RL model:

```bash
python self-edit.py \
    --experiment_name=eval_RL_iteration_1_8_epoch \
    --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch \
    --n_tasks=10 \
    --n_self_edits_per_task=5
```

### 5. Evaluate RL Iteration 1 (8 Epochs) on Eval Set

Evaluate the 8-epoch RL model on the evaluation set:

```bash
python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/eval_set_RL_iteration_1_8_epoch \
    --pretrained_checkpoint=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=9 \
    --n_self_edits=5
```

## Baseline Evaluation

### Evaluate Baseline Model

Evaluate the baseline model performance:

```bash
python eval-self-edits-baseline.py \
    --experiment_folder=${TTI_DIR}/eval_base_model \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=9
```

## SEAL RL Iteration 0

### Create Self-Edits on Eval Set (RL Iteration 1)

Generate self-edits using the first RL iteration model:

```bash
python self-edit.py \
    --experiment_name=eval_RL_iteration_1 \
    --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1 \
    --n_tasks=10 \
    --n_self_edits_per_task=5
```

## Notes

- All experiments use the Llama-3.2-1B-Instruct base model
- The experiments are designed to iteratively improve performance through self-editing and reinforcement learning
- Evaluation is performed on filtered ARC-AGI datasets for both training and evaluation sets
- LoRA (Low-Rank Adaptation) is used for efficient fine-tuning with various rank configurations
