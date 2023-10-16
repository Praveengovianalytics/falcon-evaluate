<div align="center">
  <img src="img/Falcon_Evaluate.png" alt="Falcon Evaluate Logo">
</div>

# Falcon Evaluate

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/travis/user/repo.svg)](https://github.com/Praveengovianalytics/falcon_evaluate)
[![GitHub issues](https://img.shields.io/github/issues/Praveengovianalytics/falcon-evaluate)](https://github.com/Praveengovianalytics/falcon-evaluate/issues)
[![GitHub release](https://img.shields.io/github/release/Praveengovianalytics/falcon-evaluate)](https://github.com/Praveengovianalytics/falcon-evaluate/releases)

<h4 align="center">
    <p>
        <a href="#shield-installation">Installation</a> |
        <a href="#fire-quickstart">Quickstart</a> |
    <p>
</h4>

Falcon Evaluate
==============================

Falcon Evaluate is an open-source Python library designed to simplify the process of evaluating and validating open source LLM models such as llama2,mistral ,etc. This library aims to provide an easy-to-use toolkit for assessing the performance, bias, and general behavior of LLMs in various natural language understanding (NLU) tasks.


# Falcon Evaluate - A Language Model Validation Library

## Overview
Falcon Evaluate is an open-source Python library designed to simplify the process of evaluating and validating Language Models (LLMs) such as GPT-3.5 and other similar models. This library aims to provide an easy-to-use toolkit for assessing the performance, bias, and general behavior of LLMs in various natural language understanding (NLU) tasks.

## :shield: Installation

```bash
pip install falcon_evaluate -q
```


if you want to install from source

```bash
git clone https://github.com/Praveengovianalytics/falcon_evaluate && cd falcon_evaluate
pip install -e .
```


## :fire: Quickstart

###  Google Colab notebook

- [Get start with falcon_evaluate](https://colab.research.google.com/drive/1h9E0Q5Fema9TkOiv0asyaSaHin1R0UN5?usp=sharing)

```python
# Example usage

!pip install falcon_evaluate -q

from falcon_evaluate.fevaluate_results import ModelScoreSummary
import pandas as pd
import nltk
nltk.download('punkt')

df = pd.DataFrame({
    'prompt': [
        "What is the capital of France?"
    ],
    'reference': [
        "The capital of France is Paris."
    ],
    'Model A': [
        "Paris is the capital of France.
    ],
    'Model B': [
        "Capital of France is Paris."
    ],
    'Model C': [
        "Capital of France was Paris."
    ],
})

model_score_summary = ModelScoreSummary(df)
result = model_score_summary.execute_summary()
print(result)

```

# Model Evaluation Results

The following table shows the evaluation results of different models when prompted with a question. Various scoring metrics such as BLEU score, Jaccard similarity, Cosine similarity, and Semantic similarity have been used to evaluate the models. Additionally, composite scores like Falcon Score have also been calculated.

## Evaluation Data

| Prompt                         | Reference                     |
|--------------------------------|-------------------------------|
| What is the capital of France? | The capital of France is Paris.|

## Model A Evaluation

| Response                       | Scores |
|--------------------------------|--------|
| Paris is the capital of France |  |
| **Scores**                     | **Values** |
| BLEU Score                     | 5.55e-78 |
| Jaccard Similarity             | 0.7143   |
| Cosine Similarity              | 1.0000   |
| Semantic Similarity            | 0.9628   |

### Falcon Score (Model A)

| Metric            | Value       |
|-------------------|-------------|
| Arithmetic Mean   | 0.6693      |
| Weighted Sum      | 0.6693      |
| Geometric Mean    | 4.42e-20    |
| Harmonic Mean     | 2.22e-77    |
| T-Statistic       | 1.291       |
| P-Value           | 0.2873      |
| F-Score           | 0.7692      |

## Model B Evaluation

| Response                    | Scores |
|-----------------------------|--------|
| Capital of France is Paris. |  |

### Scores

| Metric              | Value   |
|---------------------|---------|
| BLEU Score          | 0.6432  |
| Jaccard Similarity  | 0.7143  |
| Cosine Similarity   | 0.8466  |
| Semantic Similarity | 0.9954  |

### Falcon Score (Model B)

| Metric            | Value       |
|-------------------|-------------|
| Arithmetic Mean   | 0.7999      |
| Weighted Sum      | 0.7999      |
| Geometric Mean    | 0.7888      |
| Harmonic Mean     | 0.7781      |
| T-Statistic       | 0.903       |
| P-Value           | 0.4332      |
| F-Score           | 0.7692      |

## Model C Evaluation

| Response                     | Scores |
|------------------------------|--------|
| Capital of France was Paris. |  |

### Scores

| Metric              | Value       |
|---------------------|-------------|
| BLEU Score          | 9.07e-155   |
| Jaccard Similarity  | 0.5714      |
| Cosine Similarity   | 0.5803      |
| Semantic Similarity | 0.9881      |

### Falcon Score (Model C)

| Metric            | Value       |
|-------------------|-------------|
| Arithmetic Mean   | 0.5350      |
| Weighted Sum      | 0.5350      |
| Geometric Mean    | 2.34e-39    |
| Harmonic Mean     | 3.63e-154   |
| T-Statistic       | 1.178       |
| P-Value           | 0.3237      |
| F-Score           | 0.6154      |



## Key Features

1. **Benchmarking:** Falcon Evaluate provides a set of pre-defined benchmarking tasks commonly used for evaluating LLMs, including text completion, sentiment analysis, question answering, and more. Users can easily assess model performance on these tasks.

2. **Custom Evaluation:** Users can define custom evaluation metrics and tasks tailored to their specific use cases. Falcon Evaluate provides flexibility for creating custom test suites and assessing model behavior accordingly.

3. **Interpretability:** The library offers interpretability tools to help users understand why the model generates certain responses. This can aid in debugging and improving model performance.

4. **Scalability:** Falcon Evaluate is designed to work with both small-scale and large-scale evaluations. It can be used for quick model assessments during development and for extensive evaluations in research or production settings.

## Use Cases
- Model Development: Falcon Evaluate can be used during the development phase to iteratively assess and improve the performance of LLMs.
- Research: Researchers can leverage the library to conduct comprehensive evaluations and experiments with LLMs, contributing to advancements in the field.
- Production Deployment: Falcon Evaluate can be integrated into NLP pipelines to monitor and validate model behavior in real-world applications.

## Getting Started
To use Falcon Evaluate, users will need Python and dependencies such as TensorFlow, PyTorch, or Hugging Face Transformers. The library will provide clear documentation and tutorials to assist users in getting started quickly.

## Community and Collaboration
Falcon Evaluate is an open-source project that encourages contributions from the community. Collaboration with researchers, developers, and NLP enthusiasts is encouraged to enhance the library's capabilities and address emerging challenges in language model validation.

## Project Goals
The primary goals of Falcon Evaluate are to:
- Facilitate the evaluation and validation of Language Models.
- Promote transparency and fairness in AI by detecting and mitigating bias.
- Provide an accessible and extensible toolkit for NLP practitioners and researchers.

## Conclusion
Falcon Evaluate aims to empower the NLP community with a versatile and user-friendly library for evaluating and validating Language Models. By offering a comprehensive suite of evaluation tools, it seeks to enhance the transparency, robustness, and fairness of AI-powered natural language understanding systems.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── falcon_evaluate    <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io




--------