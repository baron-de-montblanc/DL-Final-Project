import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import EdgeConv, global_max_pool
from sklearn.metrics import roc_curve, auc, accuracy_score


# --------------- Define FCNN Models -----------------------


class TeacherFCNN(nn.Module):

    def __init__(self, num_features, dropout_rate=0.4):
        super(TeacherFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 1)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        # Flatten if necessary
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activ(x)
        x = self.drop2(x)

        x = torch.sigmoid(self.fc3(x))
        return x
    


class StudentFCNN(nn.Module):

    def __init__(self, num_features, dropout_rate=0.4):
        super(StudentFCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 1)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        # Flatten if necessary
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activ(x)
        x = self.drop2(x)

        x = torch.sigmoid(self.fc3(x))
        return x


# --------------- Define GNN Models -----------------------


class TeacherGNN(torch.nn.Module):
    """
    Adapted from the PHYS 2550 Hands-On Session for Lecture 21
    """
    def __init__(self):
        super(TeacherGNN, self).__init__()
        # The input feature dimension is 4 ('clus_pt', 'clus_eta', 'clus_phi', 'clus_E')
        # Ensure the MLP inside EdgeConv correctly transforms input features
        
        self.conv1 = EdgeConv(
            Sequential(
                Linear(2*4, 64),
                BatchNorm1d(64),
                ReLU(), 
                Linear(64, 64),
                BatchNorm1d(64),
                ReLU(),
                Linear(64, 64),
                BatchNorm1d(64),
                ReLU(),
            ), 
            aggr='mean')
        
        self.conv2 = EdgeConv(
            Sequential(
                Linear(64*2, 128),
                BatchNorm1d(128),
                ReLU(), 
                Linear(128, 128),
                BatchNorm1d(128),
                ReLU(),
                Linear(128, 128),
                BatchNorm1d(128),
                ReLU(),
            ), 
            aggr='mean')
        
        self.conv3 = EdgeConv(
            Sequential(
                Linear(128*2, 256),
                BatchNorm1d(256),
                ReLU(), 
                Linear(256, 256),
                BatchNorm1d(256),
                ReLU(),
                Linear(256, 256),
                BatchNorm1d(256),
                ReLU(),
            ), 
            aggr='mean')
        
        self.fc1 = Linear(256, 256)
        self.dropout = nn.Dropout(0.1)
        self.out = Linear(256, 2)


    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_max_pool(x, data.batch)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.out(x)

        return x



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
    all_labels, all_preds = [], []

    for data in train_loader:
        data, target, weights = data.to(device), data.y.to(device), data.weight.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = criterion(output, target)
        weighted_loss = (loss * weights).mean()  # Apply weights and average

        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item()

        _, preds = torch.max(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_acc = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_acc



def test_gnn(model, device, test_loader, criterion):
    model.eval()

    total_loss = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for data in test_loader:
            data, target, weights = data.to(device), data.y.to(device), data.weight.to(device)
            
            output = model(data)

            loss = criterion(output, target)
            weighted_loss = (loss * weights).mean()  # Apply weights and average

            total_loss += weighted_loss.item()

            _, preds = torch.max(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss/len(test_loader)
    avg_acc = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_acc