import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.sparse
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix
import matplotlib.pyplot as plt

# Graph SAGE

class SAGE_MLC(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16):
        super(SAGE_MLC, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

def load_data(device):
    # Load data, assuming the paths are correct
    X = torch.load('results/X_32.pt')
    Y = torch.load('results/Y.pt')
    A = scipy.sparse.load_npz('results/A/A_final.npz')

    edge_index, edge_weight = from_scipy_sparse_matrix(A)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Convert X and Y to torch tensors if they are numpy arrays
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float).to(device)
    else:
        X = X.to(device)

    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float).to(device)  # Ensuring Y is also a float for BCEWithLogitsLoss
    else:
        Y = Y.to(device)

    return X, Y, edge_index, edge_weight

def prepare_masks(num_nodes, test_size):
    # Split data into training and a temporary set first
    train_index, temp_index = train_test_split(np.arange(num_nodes), test_size=test_size, random_state=42)
    # Split the temporary set into validation and test sets
    val_index, test_index = train_test_split(temp_index, test_size=0.6667, random_state=42)  # Adjusted test_size to split remaining 30% into 20% and 10%

    # Create boolean masks for train, validation, and test datasets
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(train_index), True)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(val_index), True)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(test_index), True)

    return train_mask, val_mask, test_mask


def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(data)[mask])
        preds = (preds > 0.5).float()
        correct = (preds == data.y[mask]).float()
        accuracy = correct.mean()
    return accuracy



def save_results(losses, val_accs, file_path):
    # Convert the lists of losses and validation accuracies to CPU if they are still on CUDA device
    if isinstance(losses[0], torch.Tensor):
        losses = [loss.item() for loss in losses]  # Converts each tensor to a scalar and ensures it's not on CUDA
    if isinstance(val_accs[0], torch.Tensor):
        val_accs = [val_acc.item() for val_acc in val_accs]  # Same conversion for validation accuracies

    # Create DataFrame from lists
    df = pd.DataFrame({
        'Loss': losses,
        'Validation Accuracy': val_accs
    })

    # Save DataFrame to CSV
    df.to_csv(f'{file_path}.csv', index=False)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Assuming load_data() properly loads and converts data
    X, Y, edge_index, edge_weight = load_data(device)

    data = Data(x=X, y=Y, edge_index=edge_index, edge_attr=edge_weight).to(device)
    num_features = X.size(1)
    num_classes = Y.size(1)  # Should be number of label features

    model = SAGE_MLC(num_features, num_classes, hidden_channels=16).to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()  # Ensure BCEWithLogitsLoss is used

    test_size = 0.2
    train_mask, val_mask, test_mask = prepare_masks(Y.size(0), test_size)

    losses, val_accs = [], []
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, val_mask)

        losses.append(loss)
        val_accs.append(val_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

    test_acc = evaluate(model, data, test_mask)
    print(f'Test Accuracy: {test_acc:.4f}')

    save_results(losses, val_accs, f'results_SAGE_{test_size}')

if __name__ == "__main__":
    main()