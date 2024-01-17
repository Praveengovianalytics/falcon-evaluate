import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class Machine_ethics_evaluator:
    """
    A class to evaluate text inputs for ethical considerations using machine learning models.
    
    Attributes:
    -----------
    ethics_model_name : str
        Name of the ethical evaluation classification model.
    
    Methods:
    --------
    evaluate(df: pd.DataFrame, ethics_check: bool) -> pd.DataFrame:
        Evaluates the ethical considerations of text inputs in all columns of the given DataFrame based on specified flags.
    """
    
    def __init__(self):
        """
        Initializes the EthicsEvaluator with a specific model.
        """
        self.ethics_model_name = "Tomhuu/ethical-machine-finetuned-BERT"
        
        # Load models and tokenizers
        self.ethics_model = AutoModelForSequenceClassification.from_pretrained(self.ethics_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ethics_model_name) 

    def _classify(self, model, text):
        """
        Classify text using the given model and return probabilities for each class.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        # Remove 'token_type_ids' for models like DistilBERT that don't support it
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Mapping probabilities to respective labels
        labels = ['Positive', 'Negative', 'Neutral']
        label_probabilities = {label: round(prob, 2) for label, prob in zip(labels, probabilities)}
        return label_probabilities

    def evaluate(self, df, ethics_check=True):
        """
        Evaluates the ethical considerations of text inputs in all columns of the given DataFrame based on specified flags.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")
        if not isinstance(ethics_check, bool):
            raise ValueError("ethics_check flag must be a boolean")

        # Validate the content of each column
        for column in df.columns:
            if df[column].isnull().any():
                raise ValueError(f"Column {column} contains null values")
            if not all(isinstance(item, str) for item in df[column]):
                raise ValueError(f"Column {column} must only contain string values")

        if ethics_check:
            for column in df.columns:
                eval_results = []
                for text in df[column]:
                    try:
                        eval_result = self._classify(self.ethics_model, text)
                    except Exception as e:
                        eval_result = {"Error": str(e)}
                    eval_results.append(eval_result)
                df[f'Machine_ethics_{column}'] = eval_results
        return df
