import warnings

import pandas as pd
import textstat
import torch
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (BertForSequenceClassification, BertTokenizer,
                          GPT2LMHeadModel, GPT2Tokenizer,GPT2Config)

from .context_relevancy import FalconScoreContextRelevancy

warnings.filterwarnings("ignore")


class TextMetricsCalculator:
    def __init__(self):
        # Initializing tokenizers and models to be used
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.bert_tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
        self.bert_model = BertForSequenceClassification.from_pretrained(
            "unitary/toxic-bert"
        )

    def calculate_perplexity(self, text):
        # Tokenizing input text and calculating perplexity
        inputs = self.gpt2_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.item()

    def calculate_ari(self, text):
        # Calculating Automated Readability Index
        return textstat.automated_readability_index(text)

    def calculate_fk_grade_level(self, text):
        # Calculating Flesch-Kincaid Grade Level
        return textstat.flesch_kincaid_grade(text)

    def calculate_toxicity(self, text):
        # Tokenizing input text and calculating toxicity
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        toxicity = probs[
            0, 1
        ].item()  # Assuming that the second class is the 'toxic' class
        return toxicity

    def compute_metrics(self, row):
        # Calculating all metrics for a given row
        metrics = {
            "Perplexity": self.calculate_perplexity(row["model_generated_output"]),
            "ARI": self.calculate_ari(row["model_generated_output"]),
            "Toxicity Level": self.calculate_toxicity(row["model_generated_output"]),
            "Flesch-Kincaid Grade Level": self.calculate_fk_grade_level(
                row["model_generated_output"]
            ),
        }
        return pd.Series(metrics)


class FalconEvaluator:
    """
    FalconEvaluator class used to evaluate text data based on different metrics and context relevancy scoring methods.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the FalconEvaluator with a dataframe.
        Parameters:
        df (pd.DataFrame): A dataframe containing the data to be evaluated.
        """
        self.df = df
        self.text_metrics_calculator = TextMetricsCalculator()

    def bleu_score(self) -> float:
        """
        Calculate the BLEU score between candidate and reference texts.
        Returns:
        float: The BLEU score.
        """

        reference = [word_tokenize(self.reference)]
        candidate = word_tokenize(self.candidate)

        return sentence_bleu(reference, candidate)

    def jaccard_similarity(self) -> float:
        """
        Calculate the Jaccard similarity between candidate and reference texts.
        Returns:
        float: The Jaccard similarity score.
        """
        mlb = MultiLabelBinarizer()
        reference_tokens = set(word_tokenize(self.reference))
        model_output_tokens = set(word_tokenize(self.candidate))

        binary_reference = mlb.fit_transform([reference_tokens])
        binary_model_output = mlb.transform([model_output_tokens])

        return jaccard_score(
            binary_reference[0], binary_model_output[0], average="binary"
        )

    def cosine_similarity(self) -> float:
        """
        Calculate the cosine similarity between the TF-IDF vectors of the candidate and reference texts.
        Returns:
        float: The cosine similarity score.
        """
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([self.candidate, self.reference])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def semantic_similarity(self) -> float:
        """
        Calculate the semantic similarity between candidate and reference texts using BERT embeddings.

        Returns:
        float: The semantic similarity score.
        """
        model = SentenceTransformer("bert-base-nli-mean-tokens")
        embeddings = model.encode(
            [self.candidate, self.reference], convert_to_tensor=True
        )
        return util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0].item()

    def calculate_reference_scores(self, model_output, reference):
        reference_tokens = [word_tokenize(reference)]
        model_output_tokens = word_tokenize(model_output)
        reference_score = sentence_bleu(reference_tokens, model_output_tokens)
        return reference_score

    def calculate_precision(self, model_output, reference):
        reference_tokens = set(word_tokenize(reference))
        model_output_tokens = set(word_tokenize(model_output))

        TP = len(reference_tokens.intersection(model_output_tokens))
        FP = len(model_output_tokens) - TP
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        return precision

    def calculate_recall(self, model_output, reference):
        reference_tokens = set(word_tokenize(reference))
        model_output_tokens = set(word_tokenize(model_output))
        TP = len(reference_tokens.intersection(model_output_tokens))
        FN = len(reference_tokens) - TP
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        return recall

    def calculate_text_metrics(self, model_output):
        """
        Calculate text metrics using TextMetricsCalculator.
        Parameters:
        model_output (str): The output text from the model.
        Returns:
        dict: A dictionary containing the calculated text metrics.
        """
        row = pd.Series(
            {"model_generated_output": model_output}
        )  # Create a series to pass to compute_metrics
        metrics_series = self.text_metrics_calculator.compute_metrics(row)
        return metrics_series.to_dict()

    def calculate_falcon_score(self, model_output, reference, categories_weights, use_relevance):
        """
        Calculates and aggregates multiple evaluation metrics along with Falcon scores.

        Parameters:
        model_output (str): The output text from the model.
        reference (str): The reference text to compare against.
        weights (list): The weights for each score.

        Returns:
        tuple: A tuple containing dictionaries of individual scores and aggregated Falcon scores.
        """
        self.candidate = model_output
        self.reference = reference

        # Get the new text metrics
        text_metrics = self.calculate_text_metrics(model_output)

        # Organize metrics into specified categories
        categories = {
            "Readability and Complexity": {
                "ARI": text_metrics.get("ARI"),
                "Flesch-Kincaid Grade Level": text_metrics.get(
                    "Flesch-Kincaid Grade Level"
                ),
            },
            "Language Modeling Performance": {
                "Perplexity": text_metrics.get("Perplexity")
            },
            "Text Toxicity": {"Toxicity Level": text_metrics.get("Toxicity Level")},
            "Text Similarity and Relevance": {
                "BLEU": self.bleu_score(),
                "Cosine Similarity": self.cosine_similarity(),
                "Semantic Similarity": self.semantic_similarity(),
                "Jaccard Similarity": self.jaccard_similarity(),
            },
            "Information Retrieval": {
                "Precision": self.calculate_precision(self.candidate, self.reference),
                "Recall": self.calculate_recall(self.candidate, self.reference),
                "F1-Score": (
                                    2
                                    * self.calculate_precision(self.candidate, self.reference)
                                    * self.calculate_recall(self.candidate, self.reference)
                            )
                            / (
                                    self.calculate_precision(self.candidate, self.reference)
                                    + self.calculate_recall(self.candidate, self.reference)
                            )
                if (
                           self.calculate_precision(self.candidate, self.reference)
                           + self.calculate_recall(self.candidate, self.reference)
                   )
                   != 0
                else 0,
            },
        }
        falcon_scores_by_category = {}

        if use_relevance:
            reference_scores = self.calculate_reference_scores(
                self.candidate, self.reference
            )

            falcon = FalconScoreContextRelevancy(
                [value for category in categories.values() for value in category.values()]
            )

            for category_name, metrics in categories.items():
                falcon = FalconScoreContextRelevancy(list(metrics.values()))
                falcon_scores_by_category[category_name] = {
                    "Arithmetic Mean": falcon.arithmetic_mean(),
                    "Weighted Sum": falcon.weighted_sum(categories_weights[category_name]),
                    "Geometric Mean": falcon.geometric_mean(),
                    "Harmonic Mean": falcon.harmonic_mean(),
                    "T-Statistic": falcon.t_statistic(reference_scores),
                    "P-Value": falcon.p_value(reference_scores),
                    "F-Score": falcon.f_score(
                        metrics.get("Precision", 0), metrics.get("Recall", 0)
                    ),
                    "Z-Score Normalization": falcon.z_score_normalization(),
                }
        return categories, falcon_scores_by_category

    def evaluate(self, use_relevance):
        results = []

        for index, row in self.df.iterrows():
            prompt = row["prompt"]
            reference = row["reference"]
            evaluation_row = {"prompt": prompt, "reference": reference}
            # weights = [0.15, 0.15, 0.15, 0.15,0.10,0.10,0.10,0.10]  # customize your weights here

            categories_weights = {
                "Readability and Complexity": [0.5, 0.5],
                "Language Modeling Performance": [1],
                "Text Toxicity": [1],
                "Text Similarity and Relevance": [0.25, 0.25, 0.25, 0.25],
                "Information Retrieval": [0.33, 0.33, 0.34],
            }

            for model in self.df.columns[
                         2:
                         ]:  # Assuming model columns start from index 2
                model_output = row[model]
                scores_dict, falcon_score = self.calculate_falcon_score(
                    model_output, reference, categories_weights, use_relevance
                )
                evaluation_row[model] = self.candidate
                self.candidate = model_output
                self.reference = reference
                evaluation_row[model + "-Scores"] = scores_dict
                evaluation_row[model + "-falcon_Score"] = falcon_score

            results.append(evaluation_row)

        return pd.DataFrame(results)
