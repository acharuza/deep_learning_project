import numpy as np

class SoftVotingClassifier:

    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers
        self.weights = weights if weights is not None else [1] * len(classifiers)
        if self.weights:
            self.weights = np.array(self.weights)
            self.weights /= self.weights.sum()

    def predict_proba(self, X):
        weighted_probs = np.zeros((X.shape[0], len(self.classifiers[0].classes_)))

        for clf, weight in zip(self.classifiers, self.weights):
            probs = clf.predict_proba(X)
            weighted_probs += weight * probs

        return weighted_probs / len(self.classifiers)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)