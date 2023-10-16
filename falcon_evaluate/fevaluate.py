import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from context_relevency import FalconScoreContextRelevancy
import warnings
warnings.filterwarnings("ignore")


class FalconEvaluator:
    """
    FalconEvaluator class used to evaluate text data based on different metrics and context relevancy scoring methods.
    """

    def __init__(self, df:pd.DataFrame):

        """
        Initializes the FalconEvaluator with a dataframe.
        Parameters:
        df (pd.DataFrame): A dataframe containing the data to be evaluated.
        """
        self.df = df

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

        return jaccard_score(binary_reference[0], binary_model_output[0], average='binary')



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
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings = model.encode([self.candidate, self.reference], convert_to_tensor=True)
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


    def calculate_falcon_score(self, model_output, reference,weights):
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
        scores = [self.bleu_score(), self.jaccard_similarity(),self.cosine_similarity(), self.semantic_similarity()]
        scores_dict = {
            "bleu_score": self.bleu_score(),
            "jaccard_similarity": self.jaccard_similarity(),
            "cosine_similarity": self.cosine_similarity(),
            "semantic_similarity": self.semantic_similarity()
        }
        reference_scores = self.calculate_reference_scores(self.candidate, self.reference)
        precision = self.calculate_precision(self.candidate, self.reference)
        recall = self.calculate_recall(self.candidate, self.reference)

        falcon = FalconScoreContextRelevancy(scores)
        falcon_score = {
            'Arithmetic Mean': falcon.arithmetic_mean(),
            'Weighted Sum': falcon.weighted_sum(weights),
            'Geometric Mean': falcon.geometric_mean(),
            'Harmonic Mean': falcon.harmonic_mean(),
            'T-Statistic': falcon.t_statistic(reference_scores),  # Define reference_scores
            'P-Value': falcon.p_value(reference_scores),  # Define reference_scores
            'F-Score': falcon.f_score(precision, recall),  # Define precision and recall
            'Z-Score Normalization': falcon.z_score_normalization()
        }
        return scores_dict,falcon_score

    def evaluate(self):
        results = []

        for index, row in self.df.iterrows():
            prompt = row['prompt']
            reference = row['reference']
            evaluation_row = {'prompt': prompt, 'reference': reference}
            weights = [0.25, 0.25, 0.25, 0.25]  # customize your weights here
            for model in self.df.columns[2:]:  # Assuming model columns start from index 2
                model_output = row[model]
                scores_dict,falcon_score = self.calculate_falcon_score(model_output, reference,weights)
                evaluation_row[model] = self.candidate
                self.candidate = model_output
                self.reference = reference
                evaluation_row[model+'-Scores'] = scores_dict
                evaluation_row[model+'-falcon_Score'] = falcon_score

            results.append(evaluation_row)

        return pd.DataFrame(results)