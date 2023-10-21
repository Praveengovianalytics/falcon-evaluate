import pandas as pd


class MetricsAggregator:
    """
    MetricsAggregator class used to aggregate metrics from DataFrame columns that contain dictionaries.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input DataFrame from which the metrics will be aggregated.

    Methods
    -------
    aggregate()
        Aggregates metrics contained within dictionaries in DataFrame columns.
    """

    def __init__(self, dataframe):
        """
        Initializes the MetricsAggregator object with a DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input DataFrame containing metrics to be aggregated.
        """
        self.dataframe = dataframe

    def aggregate(self):
        """
        Aggregates metrics contained within dictionaries in DataFrame columns ending with '-Scores'.

        The aggregation is done by calculating the mean of each metric in the dictionaries.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each cell in the '-Scores' columns contains a dictionary of aggregated metrics.
        """
        # Creating an empty DataFrame to store the aggregated metrics.
        aggregated_metrics_df = pd.DataFrame()

        # Iterating over each column in the input DataFrame.
        for column in self.dataframe.columns:
            if "-Scores" in column:  # Selecting columns that end with '-Scores'.
                aggregated_metrics = (
                    {}
                )  # Dictionary to store intermediate aggregation results.

                # Iterating over each row in the selected column.
                for index, row in self.dataframe.iterrows():
                    record = row[
                        column
                    ]  # Getting the cell value (dictionary of metrics).

                    if pd.isna(record):  # Ignoring NaN or None values.
                        continue

                    # Iterating over categories and metrics in the dictionary.
                    for category, metrics in record.items():
                        if category not in aggregated_metrics:
                            aggregated_metrics[category] = {}

                        for metric, value in metrics.items():
                            if metric not in aggregated_metrics[category]:
                                aggregated_metrics[category][metric] = []

                            # Storing metric values for later aggregation.
                            aggregated_metrics[category][metric].append(value)

                # Calculating the mean of each metric.
                for category, metrics in aggregated_metrics.items():
                    for metric, values in metrics.items():
                        aggregated_metrics[category][metric] = sum(values) / len(values)

                # Storing the aggregated metrics in the result DataFrame.
                aggregated_metrics_df[column] = [aggregated_metrics]

        return aggregated_metrics_df  # Returning the DataFrame containing aggregated metrics.
