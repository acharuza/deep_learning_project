import torch
import torch.nn as nn
import torch.nn.init as init

class AlexNet(nn.Module):
    '''
        Implementation of a modified AlexNet architecture for 32x32 images
        (based on the original AlexNet: Krizhevsky et al., 2012, ImageNet Classification with Deep Convolutional Neural Networks)

        Changes made to the original architecture:
            - adjusted input size to 32x32 instead of 227x227
            - removed one fully connected layer (originally three, now two)
            - changed first convolutional layer: smaller kernel size and no large stride
            - replaced Local Response Normalization (LRN) with Batch Normalization
            - added dropout layers for regularization experiments
            - made number of output classes fixed to 10
            - added option to select weight initialization type ('random' or 'he')
        '''

    def __init__(self, dropout_rate, init_type):
        super(AlexNet, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 10)  # 10 klas na stałe
        )

        self._initialize_weights(init_type)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'random':
                    init.normal_(m.weight, mean=0, std=0.01)
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if init_type == 'he':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'random':
                    init.normal_(m.weight, mean=0, std=0.01)
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
