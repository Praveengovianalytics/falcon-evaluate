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
        Evaluates the ethical considerations of text prompts in the given DataFrame based on specified flags.
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

        Parameters:
        -----------
        model : transformers.AutoModelForSequenceClassification
            The model to use for classification.
        text : str
            The text to classify.

        Returns:
        --------
        dict
            A dictionary with the probabilities of each class (Positive, Negative, Neutral).
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
        Evaluates the ethical considerations of text prompts in the given DataFrame based on specified flags.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing text prompts to evaluate.
        ethics_check : bool
            Flag to perform ethical evaluation.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'Ethics_eval' column containing evaluation results as dictionaries.
        """
        eval_results = []
        for _, row in df.iterrows():
            prompt = row['prompt']
            result = {}
            if ethics_check:
                result['Machine_ethics_evaluation'] = self._classify(self.ethics_model, prompt)
            eval_results.append(result)
        
        df['Ethics_eval'] = eval_results
        return df