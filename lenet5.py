import torch.nn as nn
import torch


class LeNet5(nn.Module):
    '''
    Implementation of LeNet-5 architecture for CINIC-10 dataset (32x32x3)
    (based on original LeNet-5 architecture: LeCun et al., 1998, Gradient-Based Learning Applied to Document Recognition)

    Article used during the implementation:
    https://arnabfly.github.io/arnab_blog/lenet5/
    
    Changes made to the original architecture:
        - changed number of input channels to 3 (for CINIC-10 dataset)
        - changed activation functions to ReLU
        - changed pooling layers to max pooling
        - added dropout layers for regularization experiments
    '''
    

    def __init__(self, dropout_rate, init_type):
        super(LeNet5, self).__init__()

        # activation function
        self.relu = nn.ReLU()

        # Convolutional layers
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected convolutional layer with 120 feature maps each of size 1x1
        # so we can just flatten the output of s4 and use fully connected layer
        self.flatten = nn.Flatten()
        self.c5 = nn.Linear(16*5*5, 120)

        # dropout for regularization experiments
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.f6 = nn.Linear(120, 84)

        # dropout for regularization experiments
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 10 classes in CINIC-10
        self.output = nn.Linear(84, 10)

        self.init_weights(init_type)


    def init_weights(self, init_type):
        '''
        Initialize weights of the model using He initialization or random initialization (for regularization experiments)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_type == 'he':
                    nn.init.kaiming_uniform_(m.weight)
                elif init_type == 'random':
                    nn.init.uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.relu(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.relu(x)
        if self.dropout1:
            x = self.dropout1(x)
        x = self.f6(x)
        x = self.relu(x)
        if self.dropout2:
            x = self.dropout2(x)
        x = self.output(x)
        return x


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output, 1)
        return predicted
    
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            output = self(x)
            probabilities = torch.softmax(output, dim=1)
        return probabilities