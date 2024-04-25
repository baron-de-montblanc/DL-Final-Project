import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, global_max_pool


# --------------- Define FCNN Models -----------------------


class TeacherFCNN(nn.Module):

    def __init__(self, num_features, dropout_rate=0.5):
        super(TeacherFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 1)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        # Flatten if necessary
        x = x.view(x.size(0), -1)

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


class TeacherGNN(torch.nn.Module):
    """
    Adapted from the PHYS 2550 Hands-On Session for Lecture 21
    """
    def __init__(self):
        super(TeacherGNN, self).__init__()
        # The input feature dimension is 4 ('clus_pt', 'clus_eta', 'clus_phi', 'clus_E')
        # Ensure the MLP inside EdgeConv correctly transforms input features
        self.conv1 = EdgeConv(Sequential(Linear(2*4, 64), ReLU(), Linear(64, 64)), aggr='max')
        self.conv2 = EdgeConv(Sequential(Linear(64*2, 128), ReLU(), Linear(128, 128)), aggr='max')
        self.fc1 = Linear(128, 64)
        self.out = Linear(64, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print("Input x shape:", x.shape)
        x = F.relu(self.conv1(x, edge_index))
        #print("After conv1 shape:", x.shape)
        x = F.relu(self.conv2(x, edge_index))
        #print("After conv2 shape:", x.shape)
        x = global_max_pool(x, data.batch)
        #print("After global max pool shape:", x.shape)
        x = F.relu(self.fc1(x))
        #print("After fc1 shape:", x.shape)
        x = self.out(x)
        # print("Output shape:", x.shape)  # Debug output shape
        return F.sigmoid(x)



class StudentGNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO: Initialize layers and hyperparameters
        pass

    def forward(self, x):

        # TODO: initiate one forward pass
        pass


# ------------------------- Define training and testing loops (FCNN) ------------------


def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data, target, weights in train_loader:
        data, target, weights = data.to(device), target.to(device), weights.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)  # flatten the output

        loss = criterion(output, target.float())
        weighted_loss = (loss * weights).mean()  # Apply weights and average
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        
        # Manually calculate binary accuracy
        predicted = (output >= 0.5).float()
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss/len(train_loader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc



def test(model, device, test_loader, criterion):
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
      for data, target, weights in test_loader:
        data, target, weights = data.to(device), target.to(device), weights.to(device)

        output = model(data)
        output = output.squeeze(1)  # flatten the output
        loss = criterion(output, target.float())
        weighted_loss = (loss * weights).mean()

        total_loss += weighted_loss.item()
        
        # Manually calculate binary accuracy
        predicted = (output >= 0.5).float()
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss/len(test_loader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc



# ------------------------- Define training and testing loops (GNN) ------------------


def train_one_epoch_gnn(model, device, train_loader, optimizer, criterion):
    model.train()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data in train_loader:
        data = data.to(device)
        weights = data.weight
        target = torch.tensor(data.y, dtype=torch.float32, device=device)
        optimizer.zero_grad()

        output = model(data)
        output = output.squeeze(1)  # flatten the output

        loss = criterion(output, target)
        weighted_loss = (loss * weights).mean()  # Apply weights and average
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()

        # Manually calculate binary accuracy
        predicted = (output >= 0.5).float()
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss/len(train_loader)
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc



def test_gnn(model, device, test_loader, criterion):
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
      for data in test_loader:
        data = data.to(device)
        weights = data.weight
        target = torch.tensor(data.y, dtype=torch.float32)
        output = model(data)

        output = output.squeeze(1)  # flatten the output
        loss = criterion(output, target)
        weighted_loss = (loss * weights).mean()  # Apply weights and average

        total_loss += weighted_loss.item()

        # Manually calculate binary accuracy
        predicted = (output >= 0.5).float()
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss/len(test_loader)
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc