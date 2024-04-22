import numpy as np
import torch
import torch.nn as nn


# --------------- Define FCNN Models -----------------------


class TeacherFCNN(nn.Module):

    def __init__(self, dropout_rate=0.4):
        super(TeacherFCNN, self).__init__()
        self.fc1 = nn.Linear(19, 256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 1)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.activ(x)
        x = self.drop2(x)

        x = torch.sigmoid(self.fc3(x))
        return x
    


class StudentFCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(StudentFCNN, self).__init__()

        # TODO: Initialize layers and hyperparameters
        pass

    def forward(self, x):

        # TODO: initiate one forward pass
        pass


# --------------- Define GNN Models -----------------------


class TeacherGNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO: Initialize layers and hyperparameters
        pass

    def forward(self, x):

        # TODO: initiate one forward pass
        pass



class StudentGNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO: Initialize layers and hyperparameters
        pass

    def forward(self, x):

        # TODO: initiate one forward pass
        pass
