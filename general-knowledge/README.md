# SEAL - general-knowledge

This document provides instructions on how to reproduce the SEAL experiments for the *general knowledge incorporation* setting. In this setting, the goal is to update or integrate new information from a passage into the language model's weights.

The experiments involve several steps, including creating synthetic data, running a training and evaluation server, and performing Reinforcement Learning (RL) training. By following this guide, you will be able to reproduce the results from the paper.

## Prerequisites

Before you begin, please ensure that you have the following prerequisites:

*   **SLURM:** These experiments are designed to be run on a SLURM cluster. You will need to have access to a SLURM cluster and be familiar with submitting jobs using `sbatch`.
*   **ZMQ:** The experiments use [ZMQ](https://zeromq.org/) for communication between the different components of the system. Please ensure that ZMQ is installed on your system.
*   **Python dependencies:** Please make sure that you have installed all the Python dependencies listed in the `requirements.txt` file in the root of the project.

## Usage

The python files in src/ have documentation on function. Here is some information on how to run the pipelines used in the paper's experiments.

### 1. Create Data
Use `make_squad_data.sh` (or `make_squad_data_openai.sh`) to create the synthetic data used in subsequent RL training or evaluation.

```bash
sbatch general-knowledge/scripts/make_squad_data.sh
```

### 2. TTT server
Run the `TTT_server`. This sets up a [ZMQ](https://zeromq.org/) port that takes input parameters like training data and corresponding questions, and then runs rounds of training a temporary lora adapter and evaluating on the questions. This is then called for both RL training rewards and evaluation.

```bash
sbatch general-knowledge/scripts/TTT_server.sh
```

### 3. Query server
To query the server, run either `query_server` or `CPT` for either the single-passage or multi-passage setting respectively. This can be set to run on training documents for a round of ReST-EM RL training, or on validation documents for evaluation. 

```bash
sbatch general-knowledge/scripts/query_server.sh
```

### 4. RL Training
To run a round of ReST-EM, after running `query_server` on training documents, build the SFT dataset (more documentation in the python file):

```bash
python3 general-knowledge/src/EM/build_SFT_dataset.py <path/to/result/of/run.json>
```

Then, run the training script on this dataset:

```bash
sbatch general-knowledge/scripts/train_SFT.sh
```

### 5. Continual Self-Edits
To run the continual self-edits experiment (Section 5):

```bash
sbatch general-knowledge/scripts/continual_self_edits.sh
```
