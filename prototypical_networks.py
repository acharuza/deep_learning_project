import torch
from tqdm import tqdm
from easyfsl.samplers import TaskSampler
from easyfsl.utils import sliding_average
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import numpy as np
import random
import pandas as pd


class ProtoNet(nn.Module):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for data_item in data_loader:
            support_images, support_labels, query_images, query_labels, _ = data_item
            scores = (
                torch.max(
                    model(support_images.to(device), support_labels.to(device), query_images.to(device))
                    .detach()
                    .data,
                    1
                    )[1] == query_labels.to(device)
                ).sum().item(), len(query_labels)
            correct_predictions += scores[0]
            total_predictions += scores[1]

    return correct_predictions / total_predictions
    
    
def train_protonet(model, train_loader, lr=0.001, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_tqdm = tqdm(train_loader, desc="Training", leave=True)
    log_update_frequency = 10
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    loss_history = []
    
    for episode_idx, data_item in enumerate(train_tqdm):
        support_images, support_labels, query_images, query_labels, _ = data_item
        optimizer.zero_grad()
        scores = model(support_images.to(device), support_labels.to(device), query_images.to(device))
        loss = criterion(scores, query_labels.to(device))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if episode_idx % log_update_frequency == 0:
            train_tqdm.set_postfix(loss=sliding_average(loss_history, log_update_frequency))
    return loss_history
    

def create_data_loader(data_dir, n_way, n_shot, n_query, n_episodes, n_tasks, data_fraction, seed=123):
    cinic_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
    cinic_std_RGB = [0.24205776, 0.23828046, 0.25874835]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean_RGB, std=cinic_std_RGB)
    ])
    np.random.seed(seed)

    data = datasets.ImageFolder(data_dir, transform=transform)

    labels = np.array([label for _, label in data.samples])

    unique_classes = np.unique(labels)
    class_indices = {cls: np.where(labels == cls)[0] for cls in unique_classes}

    subset_size_per_class = int((len(data) * data_fraction) / len(unique_classes))

    selected_indices = np.concatenate([
        np.random.choice(class_indices[cls], min(subset_size_per_class, len(class_indices[cls])), replace=False)
        for cls in unique_classes
    ])

    data_subset = torch.utils.data.Subset(data, selected_indices)
    data_subset.get_labels = lambda: [data.targets[i] for i in selected_indices]
    data_sampler = TaskSampler(data_subset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_episodes)
    data_loader = DataLoader(data_subset, batch_sampler=data_sampler, num_workers=0, pin_memory=True, collate_fn=data_sampler.episodic_collate_fn)
    return data_loader

def experiment_protonet(n_shot, n_query, n_episodes, n_tasks, data_fraction, lr=0.001, seed=123):
    train_loader = create_data_loader("cinic-10/train", n_way=10, n_shot=n_shot, n_query=n_query, n_episodes=n_episodes, data_fraction=data_fraction, seed=seed)
    test_loader = create_data_loader("cinic-10/test", n_way=10, n_shot=n_shot, n_query=n_query, n_episodes=n_tasks, data_fraction=data_fraction, seed=seed)
    feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor.fc = nn.Flatten()
    model = ProtoNet(feature_extractor)
    loss_history = train_protonet(model, train_loader, lr=lr, seed=seed)
    accuracy = evaluate_protonet(model, test_loader)
    print(f"Accuracy on test set: {accuracy}")
    pd.DataFrame({'loss': loss_history}).to_csv(f'saved_losses/protonet_{n_shot}-{n_query}-{n_episodes}-{n_tasks}-{data_fraction}_lr_{lr}_seed_{seed}.csv', index=False)
    torch.save(model.state_dict(), f'saved_models/protonet_{n_shot}-{n_query}-{n_episodes}-{n_tasks}-{data_fraction}_lr_{lr}_seed_{seed}.pth')


if __name__ == "__main__":
    # the fraction of the dataset to use (he said to use a small fraction)
    data_fraction = 0.1
    # number of examples per class in the support set
    n_shot = 10
    # number of examples per class in the query set
    n_query = 20
    # number of episodes to train on (not epochs because we use episodic training so you can enter a larger number)
    n_episodes = 40000
    # number of tasks to test on (at the end of training accuracy will be computed on these tasks)
    n_tasks = 100
    # learning rate
    lr = 0.001
    # seed for reproducibility
    seed = 123
    # keep in mind that the loss in tqdm bar will update only every 10 episodes not to slow down the training
    experiment_protonet(n_shot, n_query, n_episodes, n_tasks, data_fraction, lr, seed)