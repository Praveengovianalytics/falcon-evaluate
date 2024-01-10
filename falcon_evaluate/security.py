import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SecurityEvaluator:
    """
    A class to evaluate text inputs for security threats using machine learning models.
    
    Attributes:
    -----------
    jailbreak_model_name : str
        Name of the jailbreak classification model.
    prompt_injection_model_name : str
        Name of the prompt injection classification model.
    
    Methods:
    --------
    evaluate(df: pd.DataFrame, jailbreak_check: bool, prompt_injection_check: bool) -> pd.DataFrame:
        Evaluates the security of text prompts in the given DataFrame based on specified flags.
    """
    
    def __init__(self):
        """
        Initializes the SecurityEvaluator with specific models.
        """
        self.jailbreak_model_name = "jackhhao/jailbreak-classifier"
        self.prompt_injection_model_name = "fmops/distilbert-prompt-injection"
        
        # Load models and tokenizers
        self.jailbreak_model = AutoModelForSequenceClassification.from_pretrained(self.jailbreak_model_name)
        self.prompt_injection_model = AutoModelForSequenceClassification.from_pretrained(self.prompt_injection_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.jailbreak_model_name)  # Assuming same tokenizer for both

    def _classify(self, model, text):
        """
        Classify text using the given model.

        Parameters:
        -----------
        model : transformers.AutoModelForSequenceClassification
            The model to use for classification.
        text : str
            The text to classify.

        Returns:
        --------
        int
            The predicted class.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        # Remove 'token_type_ids' for models like DistilBERT that don't support it
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions.argmax().item()

    def evaluate(self, df, jailbreak_check=True, prompt_injection_check=True):
        """
        Evaluates the security of text prompts in the given DataFrame based on specified flags.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing text prompts to evaluate.
        jailbreak_check : bool
            Flag to perform jailbreak classification.
        prompt_injection_check : bool
            Flag to perform prompt injection classification.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'Security_eval' column containing evaluation results.
        """
        eval_results = []
        for _, row in df.iterrows():
            prompt = row['prompt']
            result = {}
            if jailbreak_check:
                result['jailbreak_score'] = self._classify(self.jailbreak_model, prompt)
            if prompt_injection_check:
                result['prompt_injection_score'] = self._classify(self.prompt_injection_model, prompt)
            eval_results.append(result)
        
        df['Security_eval'] = eval_results
        return df