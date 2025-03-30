import numpy as np
import torch.nn as nn

class SoftVotingClassifier:

    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers
        self.weights = weights if weights is not None else [1] * len(classifiers)
        if self.weights:
            self.weights = np.array(self.weights)
            self.weights /= self.weights.sum()

    def predict_proba(self, X):
        self.classifiers[0].eval()
        first_proba = nn.Softmax(self.classifiers[0](X), dim=1)
        weighted_probs = np.zeros_like(first_proba)

        for clf, weight in zip(self.classifiers, self.weights):
            clf.eval()
            probs = nn.Softmax(clf(X), dim=1)
            weighted_probs += weight * probs

        return weighted_probs / len(self.classifiers)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)