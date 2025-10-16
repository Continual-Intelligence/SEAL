# Self-Adapting Language Models

[Paper](https://arxiv.org/abs/2506.10943), [Website](https://jyopari.github.io/posts/seal)

[Adam Zweiger](https://adamzweiger.github.io/),
[Jyothish Pari](https://jyopari.github.io),
[Han Guo](https://han-guo.info/),
[Ekin Akyürek](https://ekinakyurek.github.io/),
[Yoon Kim](https://people.csail.mit.edu/yoonkim/),
[Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/)

MIT CSAIL

<img src="few-shot/assets/SEAL.png" alt="SEAL" width="400"/>

SEAL (**Se**lf-**A**dapting **L**LMs) is a framework for training language models via RL to generate self-edits (finetuning data and other update directives for themselves) in response to new inputs. 

We explore SEAL in two domains:
- [general-knowledge](general-knowledge): Incorporating new factual knowledge
- [few-shot](few-shot): Adapting to new tasks from few-shot examples

Both folders include code, data, and documentation.

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Continual-Intelligence/SEAL.git
cd SEAL
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n seal_env python=3.12
conda activate seal_env
```

Using **venv**:

```bash
python3.12 -m venv seal_env
source seal_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

To run the SEAL framework, you will need to provide an OpenAI API key. This key is used to access the GPT models that are used for generating self-edits.

Create a `.env` file in the project root and add your OpenAI API key to it:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The `.env` file is used to store environment variables that are specific to your local environment. The SEAL framework will automatically load the variables from this file.

### 5. SLURM users

Before running any shell scripts, make sure to update the SLURM directives at the top of each `.sh` file to match your system configuration. All experiments can be run with 2 A100/H100 GPUs. Other setups may require refactoring and/or changing model sizes.


## 📂 Project Structure

The project is organized as follows:

```
.
├── few-shot/
│   ├── README.md
│   ├── ...
├── general-knowledge/
│   ├── README.md
│   ├── ...
├── LICENSE
├── README.md
└── requirements.txt
```

*   `few-shot/`: This directory contains the code and data for the few-shot learning experiments.
*   `general-knowledge/`: This directory contains the code and data for the general-knowledge experiments.
*   `LICENSE`: The license for the project.
*   `README.md`: The main README file for the project.
*   `requirements.txt`: The Python dependencies for the project.


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## 📄 Citation

If you found this work useful, please cite:

```
@misc{zweiger2025selfadaptinglanguagemodels,
      title={Self-Adapting Language Models}, 
      author={Adam Zweiger and Jyothish Pari and Han Guo and Ekin Akyürek and Yoon Kim and Pulkit Agrawal},
      year={2025},
      eprint={2506.10943},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.10943}, 
}
```
