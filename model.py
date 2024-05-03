import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import EdgeConv, global_max_pool
from sklearn.metrics import roc_curve, auc, accuracy_score


# --------------- Define FCNN Model -----------------------


class JetFCNN(nn.Module):

    def __init__(self, num_features, num_hidden_layers=2, dropout_rate=0.1):
        super(JetFCNN, self).__init__()

        self.fc1 = nn.Linear(num_features, 400)
        self.bn1 = nn.BatchNorm1d(400)

        # hidden layers
        self.fch = nn.Linear(400, 400)
        self.hidden_bns = nn.ModuleList([nn.BatchNorm1d(400) for _ in range(num_hidden_layers)])

        # "constants"
        self.out = nn.Linear(400, 2)
        self.activ = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        # Initialize weights using Glorot Uniform (Xavier Uniform)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fch.weight)
        init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        # Flatten if necessary
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.drop(x)

        for i in range(len(self.hidden_bns)):
            x = self.fch(x)
            x = self.hidden_bns[i](x)
            x = self.activ(x)
            x = self.drop(x)

        x = self.out(x)
        return x
    

# --------------- Define GNN Model -----------------------


class JetGNN(torch.nn.Module):
    """
    Adapted from the PHYS 2550 Hands-On Session for Lecture 21
    + Architecture borrowed from Jet Tagging via Particle Clouds
    """
    def __init__(self):
        super(JetGNN, self).__init__()
        # The input feature dimension is 7 (preprocessed features)
        # Ensure the MLP inside EdgeConv correctly transforms input features
        
        self.conv1 = EdgeConv(
            Sequential(
                Linear(2*7, 64),
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


# ------------------------- Define training/eval function ------------------


def run_model(model, model_type, device, data_loader, criterion, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.set_grad_enabled(train):
        for batch in data_loader:
            if model_type == 'fcnn':
                data, target, weights = batch
            elif model_type == 'gnn':
                data, target, weights = batch.x, batch.y, batch.weight
            else:
                raise ValueError("Unknown model type provided")

            data, target, weights = data.to(device), target.to(device), weights.to(device)

            if train:
                optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            weighted_loss = (loss * weights).mean()

            if train:
                weighted_loss.backward()
                optimizer.step()

            total_loss += weighted_loss.item()

            # Compute probabilities from logits using softmax for the positive class
            probs = F.softmax(output, dim=1).detach().cpu().numpy()[:, 1]
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    predictions = np.array(all_probs) > 0.5
    avg_acc = accuracy_score(all_labels, predictions)

    if not train:
        # Prepare ROC curve data
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        return avg_loss, avg_acc, fpr, tpr
    
    return avg_loss, avg_acc




# --------------- Define Transfer Learning function -------------------------------------



def train_with_distillation(student_model, teacher_model, model_type, train_loader, 
                            criterion, optimizer, device, alpha=0.5, temperature=2.0):
    teacher_model.eval()
    student_model.train()

    total_loss = 0
    all_labels, all_preds = [], []

    for batch in train_loader:
        if model_type == 'fcnn':
            data, target, weights = batch
        elif model_type == 'gnn':
            data, target, weights = batch.x, batch.y, batch.weight
        else:
            raise ValueError("Unknown model type provided")

        data, target, weights = data.to(device), target.to(device), weights.to(device)

        # Teacher model's output
        with torch.no_grad():
            soft_labels = teacher_model(data)
            soft_labels = F.softmax(soft_labels / temperature, dim=1)

        # Student model's output
        outputs = student_model(data)
        soft_outputs = F.log_softmax(outputs / temperature, dim=1)

        # Calculate loss
        loss_hard = criterion(outputs, target)  # Hard label loss
        loss_soft = F.kl_div(soft_outputs, soft_labels.detach(), reduction='batchmean')  # Soft label loss
        loss = alpha * loss_hard + (1 - alpha) * temperature * temperature * loss_soft  # Total loss, scaled by temperature^2 as in Hinton's paper
        weighted_loss = (loss * weights).mean()  # Apply weights and average

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()

        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_acc = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_acc

