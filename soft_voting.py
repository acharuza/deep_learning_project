import numpy as np
import torch
import torch.nn.functional as F

class SoftVotingClassifier:

    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers
        self.weights = weights if weights is not None else [1] * len(classifiers)
        if self.weights:
            self.weights = np.array(self.weights)
            self.weights /= self.weights.sum()

    def predict_proba(self, X):
        self.classifiers[0].eval()
        first_proba = F.softmax(self.classifiers[0](X), dim=1).detach().numpy()
        weighted_probs = np.zeros_like(first_proba)
        weighted_probs = weighted_probs + self.weights[0] * first_proba

        for clf, weight in zip(self.classifiers[1:], self.weights[1:]):
            clf.eval()
            probs = F.softmax(clf(X), dim=1).detach().numpy()
            weighted_probs = weighted_probs + weight * probs

        return weighted_probs / len(self.classifiers)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    

if __name__ == "__main__":
    from alexnet import AlexNet
    from lenet5 import LeNet5

    classifiers = [AlexNet(dropout_rate=0, init_type="random"), LeNet5(dropout_rate=0, init_type="random")]
    weights = [0.5, 0.5]
    soft_voting_classifier = SoftVotingClassifier(classifiers, weights)

    # X CINIC-10 image
    X = torch.randn(1, 3, 32, 32)  # Example input tensor

    print(soft_voting_classifier.predict_proba(X))
    print(soft_voting_classifier.predict(X))
