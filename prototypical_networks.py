import torch
from tqdm import tqdm
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average


class ProtoNet:
    '''
    Implementation of the Prototypical Networks for Few Shot Learning based
    on https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb#scrollTo=KwQe3FFT3FZ0
    '''
    def __init__(self, feature_extractor):
        super(ProtoNet, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support_images, support_labels, query_images):
        features_support = self.feature_extractor(support_images)
        features_query = self.feature_extractor(query_images)

        n_way = len(torch.unique(support_labels))
        prototype = torch.cat(
            [
                features_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)
            ]
        )
        distances = torch.cdist(features_query, prototype)
        return -distances
    
def evaluate_protonet(model, data_loader):
    total_predictions = 0
    correct_predictions = 0
    model.eval()

    with torch.no_grad():
        for data_item in data_loader:
            support_images, support_labels, query_images, query_labels, _ = data_item
            scores = (
                torch.max(
                    model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
                    .detach()
                    .data,
                    1
                    )[1] == query_labels.cuda()
                ).sum().item(), len(query_labels)
            correct_predictions += scores[0]
            total_predictions += scores[1]

    return correct_predictions / total_predictions
    
def train_protonet(model, train_loader, lr=0.001):
    train_tqdm = tqdm(train_loader, desc="Training", leave=True)
    log_update_frequency = 10
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    loss_history = []
    
    for episode_idx, data_item in enumerate(train_tqdm):
        support_images, support_labels, query_images, query_labels, _ = data_item
        optimizer.zero_grad()
        scores = model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
        loss = criterion(scores, query_labels.cuda())
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if episode_idx % log_update_frequency == 0:
            train_tqdm.set_postfix(loss=sliding_average(loss_history, log_update_frequency))
    return loss_history
    

