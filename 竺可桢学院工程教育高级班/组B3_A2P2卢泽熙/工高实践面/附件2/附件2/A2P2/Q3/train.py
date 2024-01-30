# import the module you need
import numpy as np
from nn import *

# Define your model
class YourCNNModel:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dy, lr):
        pass


if __name__ == '__main__':
    # Use cv2 or pil to read and pretreat the image data
    train_data = ...
    train_label = ...
    test_data = ...
    test_label = ...

    # Define the hyperparameter of your model
    epochs = ...
    batch_size = ...
    lr = ...

    # Get the object of model and loss
    model = ...
    criterion = ...

    # You can use tqdm or so to optimize the commandline output
    
    for epoch in range(epochs):

        train_loss = 0.
        train_acc = 0.

        # Do Training
        for i in range(len(train_data) // batch_size + 1):
            input = ...
            label = ...
            ...
        
        train_loss /= len(train_data) // batch_size + 1
        train_acc /= len(train_data) // batch_size + 1

        valid_loss = 0.
        valid_acc = 0.

        # Do the validation test
        for i in range(len(test_data) // batch_size + 1):
            input = ...
            label = ...
            ...
        
        valid_loss /= len(test_data) // batch_size + 1
        valid_acc /= len(test_data) // batch_size + 1

        # Do some recording thing
        ...
