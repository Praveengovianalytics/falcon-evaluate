import itertools

import plotly.graph_objects as go


class ModelPerformancePlotter:
    """
    ModelPerformancePlotter class used to represent and plot the performance of different models based on various metrics.

    usage :- ModelPerformancePlotter(aggregated_metrics_df).get_falcon_performance_quadrant()

    ...

    Attributes
    ----------
    df : DataFrame
        a pandas DataFrame containing the performance metrics of models
    colors : cycle
        an iterator cycling through colors for model markers in the plot
    shapes : cycle
        an iterator cycling through shapes for model markers in the plot

    Methods
    -------
    combine_scores(scores_dict, keys):
        Combines scores by calculating the mean of selected metrics.

    normalize(value, min_value, max_value):
        Normalizes a value between a min_value and max_value.

    plot():
        Generates and displays a plot representing the models' performance.
    """

    def __init__(self, df):
        """
        Constructs all the necessary attributes for the ModelPerformancePlotter object.

        Parameters:
        ----------
        df : DataFrame
            a pandas DataFrame containing the performance metrics of models
        """

        self.df = df
        self.colors = itertools.cycle(
            [
                "blue",
                "red",
                "green",
                "purple",
                "orange",
                "pink",
                "yellow",
                "brown",
                "grey",
            ]
        )
        self.shapes = itertools.cycle(
            [
                "circle",
                "square",
                "diamond",
                "cross",
                "x",
                "triangle-up",
                "star",
                "hexagon",
                "octagon",
            ]
        )

    @staticmethod
    def combine_scores(scores_dict, keys):
        """
        Combines scores by calculating the mean of the values of selected metrics from a dictionary.

        Parameters:
        ----------
        scores_dict : dict
            a dictionary containing metric names as keys and their scores as values
        keys : list
            a list of selected metric names

        Returns:
        -------
        float
            the mean of the selected metric scores
        """

        combined_score = 0
        count = 0
        for key in keys:
            for metric, value in scores_dict[key].items():
                combined_score += value
                count += 1
        return combined_score / count if count != 0 else 0

    @staticmethod
    def normalize(value, min_value, max_value):
        """
        Normalizes a value between a minimum and a maximum value.

        Parameters:
        ----------
        value : float
            the value to be normalized
        min_value : float
            the minimum value for normalization
        max_value : float
            the maximum value for normalization

        Returns:
        -------
        float
            the normalized value
        """
        return (
            (value - min_value) / (max_value - min_value)
            if (max_value - min_value) != 0
            else 0
        )

    def get_falcon_performance_quadrant(self):
        """
        Generates and displays a plot representing the models' performance based on various metrics.

        Creates a scatter plot where each model is represented by a marker with customized color and shape.
        The x and y coordinates of the markers are derived from the models' performance metrics.

        """
        models = self.df.columns
        x_scores = []
        y_scores = []

        for model in models:
            scores = self.df[model].iloc[0]
            x_scores.append(
                self.combine_scores(
                    scores,
                    ["Readability and Complexity", "Language Modeling Performance"],
                )
            )
            y_scores.append(
                self.combine_scores(
                    scores, ["Text Similarity and Relevance", "Information Retrieval"]
                )
            )

        x_min, x_max = min(x_scores), max(x_scores)
        y_min, y_max = min(y_scores), max(y_scores)

        fig = go.Figure()

        for model, x_score, y_score in zip(models, x_scores, y_scores):
            norm_x_score = self.normalize(x_score, x_min, x_max)
            norm_y_score = self.normalize(y_score, y_min, y_max)

            fig.add_trace(
                go.Scatter(
                    x=[norm_x_score],
                    y=[norm_y_score],
                    mode="markers+text",
                    name=model,
                    text=[model],
                    marker=dict(
                        size=10, color=next(self.colors), symbol=next(self.shapes)
                    ),
                )
            )

        # Draw lines to create quadrants
        fig.add_shape(
            type="line",
            x0=0.5,
            y0=0,
            x1=0.5,
            y1=1,
            line=dict(
                color="Black",
                width=2,
                dash="dash",
            ),
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0.5,
            x1=1,
            y1=0.5,
            line=dict(
                color="Black",
                width=2,
                dash="dash",
            ),
        )

        # Update layout for better appearance
        fig.update_layout(
            title="Falcon - Performance Quadrant",
            xaxis_title="Readability + Language Modeling Performance",
            yaxis_title="Text Relevance + Information Retrieval",
            legend_title="Models",
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )

        # Show plot
        fig.show()
