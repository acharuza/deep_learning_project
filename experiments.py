from lenet5 import LeNet5
from model_training import train_model
from data_loading import DataLoaderCreator
import numpy as np
import pandas as pd
import torch

def default_parameters(lr, max_epochs, seed):
    '''
    Default parameters for training
    '''
    cnn = LeNet5(dropout_rate=0, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/constant_lr_{lr}_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/constant_lr_{lr}_seed_{seed}.pth')

def learning_rate_reduce(lr, max_epochs, seed):
    '''
    Reducing learning rate when reaching a plateau
    '''
    cnn = LeNet5(dropout_rate=0, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, lr_scheduling=True, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/lr_reduce_{lr}_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/lr_reduce_{lr}_seed_{seed}.pth')

def he_weight_init(lr, max_epochs, seed):
    '''
    He weight initialization
    '''
    cnn = LeNet5(dropout_rate=0, init_type='he')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/he_init_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/he_init_seed_{seed}.pth')

def dropout(lr, max_epochs, seed):
    '''
    Dropout with 20% dropout rate
    '''
    cnn = LeNet5(dropout_rate=0.2, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.standard_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/dropout_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/dropout_seed_{seed}.pth')

def random_rotation(lr, max_epochs, seed):
    '''
    Random rotation from -20 to 20 degrees
    '''
    cnn = LeNet5(dropout_rate=0, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_rotation_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/random_rotation_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/random_rotation_seed_{seed}.pth')

def random_flipping(lr, max_epochs, seed):
    '''
    Random horizontal flipping
    '''
    cnn = LeNet5(dropout_rate=0, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_flipping_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/random_flipping_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/random_flipping_seed_{seed}.pth')

def random_blur(lr, max_epochs, seed):
    '''
    Random Gaussian blur with kernel size 3
    '''
    cnn = LeNet5(dropout_rate=0, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_blur_transform, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/random_blur_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/random_blur_seed_{seed}.pth')

def random_all(lr, max_epochs, seed):
    '''
    All data augmentation techniques combined
    '''
    cnn = LeNet5(dropout_rate=0, init_type='random')
    loader_creator = DataLoaderCreator(seed)
    train_loader = loader_creator.create_data_loader('cinic-10/train', batch_size=128, transform=loader_creator.random_all, shuffle=True)
    val_loader = loader_creator.create_data_loader('cinic-10/valid', batch_size=128, transform=loader_creator.standard_transform, shuffle=False)
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr, seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/random_all_seed_{seed}.csv', index=False)
    torch.save(cnn.state_dict(), f'saved_models/random_all_seed_{seed}.pth')


if __name__ == "__main__":
    # choose the learning rate, max epochs and seed
    # then choose the experiment you want to run
    # experiments will save history of train and val losses and model weights
    lr = 0.01
    max_epochs = 50
    seed = 123
    default_parameters(lr, max_epochs, seed)