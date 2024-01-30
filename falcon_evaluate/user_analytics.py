from transformers import pipeline
import pandas as pd

class Emotions:
    """
    A class to evaluate text inputs for emotion detection using a pre-trained model.

    Attributes:
    -----------
    classifier : pipeline
        Hugging Face pipeline for emotion classification.
    positive_emotions : list
        List of emotions considered positive.
    negative_emotions : list
        List of emotions considered negative.
    neutral_emotions : list
        List of emotions considered neutral.

    Methods:
    --------
    evaluate(df: pd.DataFrame) -> pd.DataFrame:
        Evaluates the emotions of text inputs in all columns of the given DataFrame.
    """

    def __init__(self, positive_emotions=None, negative_emotions=None, neutral_emotions=None):
        """
        Initializes the Emotions class with a specific model and emotion lists.
        """

        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        # Set default emotion lists if none are provided
        self.positive_emotions = positive_emotions if positive_emotions is not None else [
            'approval', 'joy', 'caring', 'desire', 'admiration', 
            'optimism', 'love', 'excitement', 'amusement', 'gratitude', 'pride'
        ]
        self.negative_emotions = negative_emotions if negative_emotions is not None else [
            'disappointment', 'sadness', 'annoyance', 'disapproval', 'nervousness', 
            'anger', 'embarrassment', 'remorse', 'disgust', 'grief', 'confusion', 'fear'
        ]
        self.neutral_emotions = neutral_emotions if neutral_emotions is not None else [
            'neutral', 'realization', 'curiosity', 'surprise', 'relief'
        ]

    def evaluate(self, df):
        """
        Evaluates the emotions of text inputs in all columns of the given DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")

        # Validate the content of each column
        for column in df.columns:
            if df[column].isnull().any():
                raise ValueError(f"Column {column} contains null values")
            if not all(isinstance(item, str) for item in df[column]):
                raise ValueError(f"Column {column} must only contain string values")

        def format_emotion_scores(data):
            # Round scores to two decimal places and reformat into a single dictionary
            formatted_data = {"{}".format(item['label']): round(item['score'], 2) for item in data}
            return formatted_data

        def categorize_emotion_scores(emotion_scores):
            # Calculate the total score for each category
            positive_score = sum(emotion_scores.get(emotion, 0) for emotion in self.positive_emotions)
            negative_score = sum(emotion_scores.get(emotion, 0) for emotion in self.negative_emotions)
            neutral_score = sum(emotion_scores.get(emotion, 0) for emotion in self.neutral_emotions)
            
            # Determine the dominant category
            if positive_score > negative_score and positive_score > neutral_score:
                return 'Positive'
            elif negative_score > positive_score and negative_score > neutral_score:
                return 'Negative'
            else:
                return 'Neutral'

        # Apply the classifier to each text and store the results in a new column
        for column in df.columns:
            # Calculate the emotion scores
            df[f'Emotions_score_{column}'] = df[column].apply(lambda text: self.classifier(text)[0])
            # Format the emotion scores and calculate the classification
            df[f'Emotions_{column}'] = df[f'Emotions_score_{column}'].apply(lambda text: {
                'User_analytics-emotions': format_emotion_scores(text),
                'User_analytics-emotions_classification': categorize_emotion_scores(format_emotion_scores(text))
            })
            df.drop(columns=f'Emotions_score_{column}', inplace=True)
        return df


"""
Usage example:

# Example usage:
df = pd.DataFrame({'Prompt': ["I am not having a great day", "I love sunny days"],
                   'Reference': ["Its sorry to hear , Do your best", "Thats good to hear"],
                   'Model_A': ["The day is good , but you dont ok it seems", "Not all the days are sunny days"]})
emotions = Emotions()
result_df = emotions.evaluate(df)
result_df

"""