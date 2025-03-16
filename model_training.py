import torch
import torch.nn as nn
import torch.optim as optim
from early_stopping import EarlyStopper


def train_model(model, train_loader, val_loader, learning_rate, lr_scheduling, max_epochs, weight_decay, seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_scheduling:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_losses = []
    val_losses = []

    early_stopper = EarlyStopper(patience=5)

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        average_val_loss = running_val_loss / len(val_loader)

        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        if early_stopper.early_stop(average_val_loss):
            print(f'Early stopping at epoch {epoch}')
            break

        scheduler.step(running_loss)

    return train_losses, val_losses

