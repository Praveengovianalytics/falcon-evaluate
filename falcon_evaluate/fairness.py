from transformers import pipeline
import pandas as pd

class FairnessEvaluator:
    """
    A class to evaluate text inputs for stereotypes using the Hugging Face Transformers pipeline.

    Attributes:
    -----------
    nlp : transformers.Pipeline
        The pipeline for stereotype classification.

    Methods:
    --------
    evaluate(df: pd.DataFrame) -> pd.DataFrame:
        Evaluates the fairness of text prompts in the given DataFrame.
    """

    def __init__(self):
        """
        Initializes the FairnessEvaluator with a specific model.
        """
        self.compute_stereotype_score = pipeline("text-classification",
                            model="wu981526092/Sentence-Level-Stereotype-Detector",
                            tokenizer="wu981526092/Sentence-Level-Stereotype-Detector")

    def evaluate(self, df):
        """
        Evaluates the fairness of text prompts in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing text prompts to evaluate.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'Fairness_eval' column containing evaluation results.
        """
        eval_results = []
        for _, row in df.iterrows():
            prompt = row['prompt']
            result = self.compute_stereotype_score(prompt)
            eval_results.append({'stereotype_score':result})

        df['Fairness_eval'] = eval_results
        return df
