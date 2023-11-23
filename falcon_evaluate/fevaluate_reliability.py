import pandas as pd
from sentence_transformers import CrossEncoder

class Reliability_evaluator:
    """
    This class is designed to evaluate hallucination scores for multiple
    model outputs against reference sentences using a CrossEncoder model.
    """
    def __init__(self, model_name='vectara/hallucination_evaluation_model'):
        """
        Initializes the SentenceSimilarityEvaluator with a specified model.

        Parameters:
        - model_name (str): The name of the model to be used for hallucination evaluation.
        """
        self.model = CrossEncoder(model_name)

    def predict_hallucination_score(self, dataframe:pd.DataFrame):
        """
        Predicts similarity scores for each model output in the DataFrame against the reference sentences.

        Parameters:
        - dataframe (pandas.DataFrame): A DataFrame containing 'reference' and multiple model output columns.

        Returns:
        - results (pandas.DataFrame): The original DataFrame with additional columns for hallucination scores.
        """
        results = dataframe.copy()
        for column in dataframe.columns:
            if column not in ["prompt", "reference"]:
                sentence_pairs = list(zip(dataframe["reference"], dataframe[column]))
                scores = self.model.predict(sentence_pairs)
                results[f"{column}-reliability-Score"] = [{'hallucination_score': round(score,2)} for score in scores]
        return results

"""
# Example usage
if __name__ == "__main__":
    # Example DataFrame
    df = pd.DataFrame(
        {
            "prompt": [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of Italy?",
                "What is the capital of Spain?",
                "What is the capital of Portugal?",
                "What is the capital of Greece?",
                "What is the capital of Poland?",
                "What is the capital of Belgium?",
                "What is the capital of Netherlands?",
                "What is the capital of Austria?",
            ],
            "reference": [
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
                "The capital of Italy is Rome.",
                "The capital of Spain is Madrid.",
                "The capital of Portugal is Lisbon.",
                "The capital of Greece is Athens.",
                "The capital of Poland is Warsaw.",
                "The capital of Belgium is Brussels.",
                "The capital of Netherlands is Amsterdam.",
                "The capital of Austria is Vienna.",
            ],
            "Model A": [
                "Paris is the capital of France.",
                "Berlin is Germany’s capital.",
                "Rome is the capital of Italy.",
                "Madrid is the capital of Spain.",
                "Lisbon is the capital of Portugal.",
                "Athens is the capital of Greece.",
                "Warsaw is the capital of Poland.",
                "Brussels is the capital of Belgium.",
                "Amsterdam is the capital of Netherlands.",
                "Vienna is the capital of Austria.",
            ],
            "Model B": [
                "Capital of France is Paris.",
                "Germany’s capital city is Berlin.",
                "Italy's capital city is Rome.",
                "Spain's capital is Madrid.",
                "Portugal's capital is Lisbon.",
                "Capital of Greece is Athens.",
                "Poland’s capital city is Warsaw.",
                "Capital city of Belgium is Brussels.",
                "Netherlands has Amsterdam as its capital.",
                "Capital of Austria? It's Vienna.",
            ],
            "Model C": [
                "Capital of France was Paris.",
                "Germany’s capital city is not Berlin.",
                "Was Rome the capital of Italy?",
                "Madrid, Spain's capital?",
                "Is Lisbon the main city of Portugal?",
                "Athens might be the capital of Greece.",
                "Warsaw was the main city of Poland.",
                "Isn’t Brussels the heart of Belgium?",
                "Amsterdam, known as the Netherlands' capital.",
                "Vienna stands as Austria's capital.",
            ],
        }
    )

    # Instantiate the evaluator with the desired model

    Reliability_eval = Reliability_evaluator()

    # Compute hallucination scores
    results_df = Reliability_eval.predict_hallucination_score(df)

    results_df.to_csv('hallucination_scores.csv')

    # Print or further process the results
    print(results_df)
"""