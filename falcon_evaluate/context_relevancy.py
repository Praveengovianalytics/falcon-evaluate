import numpy as np
import scipy.stats as stats


class FalconScoreContextRelevancy:
    def __init__(self, scores):
        self.scores = scores

    def arithmetic_mean(self):
        return np.mean(self.scores)

    def weighted_sum(self, weights):
        return np.dot(self.scores, weights)

    def geometric_mean(self):
        return stats.gmean(self.scores)

    def harmonic_mean(self):
        return stats.hmean(self.scores)

    def t_statistic(self, reference_scores):
        t_stat, p_value = stats.ttest_ind(self.scores, reference_scores)
        return t_stat

    def p_value(self, reference_scores):
        t_stat, p_value = stats.ttest_ind(self.scores, reference_scores)
        return p_value

    def f_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def z_score_normalization(self):
        mean = np.mean(self.scores)
        std_dev = np.std(self.scores)
        z_scores = [(x - mean) / std_dev for x in self.scores]
        return z_scores
