import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, global_max_pool


# --------------- Define FCNN Models -----------------------


class TeacherFCNN(nn.Module):

    def __init__(self, num_features, dropout_rate=0.4):
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
        return F.log_softmax(x, dim=1)



class StudentGNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO: Initialize layers and hyperparameters
        pass

    def forward(self, x):

        # TODO: initiate one forward pass
        pass


# ------------------------- Define training and testing loops (FCNN) ------------------


def train_one_epoch(model, device, train_loader, optimizer, criterion, acc_metric):
    model.train()

    total_loss = 0
    accuracy_metric = acc_metric.to(device)
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)  # flatten the output
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy_metric(output, target.float())

    avg_loss = total_loss/len(train_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc



def test(model, device, test_loader, criterion, acc_metric):
    model.eval()

    total_loss = 0
    accuracy_metric = acc_metric.to(device)
    with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)

          output = model(data)
          output = output.squeeze(1)  # flatten the output
          loss = criterion(output, target.float())

          total_loss += loss.item()
          accuracy_metric(output, target.float())

    avg_loss = total_loss/len(test_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc



# ------------------------- Define training and testing loops (GNN) ------------------


def train_one_epoch_gnn(model, device, train_loader, optimizer, criterion, acc_metric):
    model.train()

    total_loss = 0
    accuracy_metric = acc_metric.to(device)
    for data in train_loader:
        data = data.to(device)
        target = torch.tensor(np.round(data.y), dtype=torch.long)  # export target to tensor
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)  # flatten the output
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy_metric(output, target)

    avg_loss = total_loss/len(train_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc



def test_gnn(model, device, test_loader, criterion, acc_metric):
    model.eval()

    total_loss = 0
    accuracy_metric = acc_metric.to(device)
    with torch.no_grad():
      for data in test_loader:
          data = data.to(device)
          target = torch.tensor(np.round(data.y), dtype=torch.long)  # export target to tensor
          output = model(data)
          output = output.squeeze(1)  # flatten the output
          loss = criterion(output, target.float())

          total_loss += loss.item()
          accuracy_metric(output, target)

    avg_loss = total_loss/len(test_loader)
    avg_acc = accuracy_metric.compute()
    return avg_loss, avg_acc