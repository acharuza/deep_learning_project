from lenet5 import LeNet5
from model_training import train_model
from data_loading import DataLoaderCreator
import numpy as np
import pandas as pd
import torch

# Fake implementation of AlexNet so that the code runs without errors
# Remove this when you implement AlexNet
class AlexNet:
    pass


def default_parameters(model, lr, max_epochs, seed):
    '''
    Default parameters for training
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_constant_lr_{lr}_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_constant_lr_{lr}_seed_{seed}.pth')

def learning_rate_reduce(model, lr, max_epochs, seed):
    '''
    Reducing learning rate when reaching a plateau
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, lr_scheduling=True, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_lr_reduce_{lr}_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_lr_reduce_{lr}_seed_{seed}.pth')

def he_weight_init(model, lr, max_epochs, seed):
    '''
    He weight initialization
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='he')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='he')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_he_init_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_he_init_seed_{seed}.pth')

def dropout(model, lr, max_epochs, seed):
    '''
    Dropout with 20% dropout rate
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0.2, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0.2, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_dropout_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_dropout_seed_{seed}.pth')

def weight_decay(model, lr, max_epochs, seed):
    '''
    Weight decay with 0.0001
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, weight_decay=0.0001, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_weight_decay_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_weight_decay_seed_{seed}.pth')

def random_rotation(model, lr, max_epochs, seed):
    '''
    Random rotation from -20 to 20 degrees
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_rotation_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_random_rotation_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_random_rotation_seed_{seed}.pth')

def random_flipping(model, lr, max_epochs, seed):
    '''
    Random horizontal flipping
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_flipping_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_random_flipping_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_random_flipping_seed_{seed}.pth')

def random_blur(model, lr, max_epochs, seed):
    '''
    Random Gaussian blur with kernel size 3
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_blur_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_random_blur_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_random_blur_seed_{seed}.pth')

def random_all(model, lr, max_epochs, seed):
    '''
    All data augmentation techniques combined
    '''
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_all, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_random_all_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_random_all_seed_{seed}.pth')

def cut_mix(model, lr, max_epochs, seed):
    if model == "lenet5":
        cnn = LeNet5(dropout_rate=0, init_type='random')
    elif model == "alexnet":
        cnn = AlexNet(dropout_rate=0, init_type='random')
    else:
        raise ValueError("Invalid model name")
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader_cutmix('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model}_cut_mix_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/{model}_cut_mix_seed_{seed}.pth')

if __name__ == "__main__":
    # choose the learning rate, max epochs and seed
    # then choose the experiment you want to run
    # experiments will save history of train and val losses and model weights
    model = "lenet5"
    lr = 0.01
    max_epochs = 50
    seed = 123
    cut_mix(model, lr, max_epochs, seed)