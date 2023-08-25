import os
import random
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from pyarrow import dataset as ds
from torch_geometric.utils import negative_sampling
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load parquet data
def load_parquet_data(path):
    table = pq.read_table(path)
    return table.to_pandas()


#Graph creation with ChemBL IDs as nodes and linked diseases and targets as edges
def create_graph_from_parquet(data_path):
    dataset = ds.dataset(data_path, format="parquet")
    
    edge_sources_indices = []
    edge_targets_diseases_indices = []
    edge_targets_targets_indices = []
    for batch in dataset.to_table().to_batches():
        for molecule_id, linked_diseases, linked_targets in zip(batch.column('id'), batch.column('linkedDiseases'), batch.column('linkedTargets')):
            
            # Handle linked_diseases
            diseases = linked_diseases.as_py().get('rows', []) if linked_diseases and linked_diseases.as_py() else []
            edge_sources_indices.extend([molecule_id.as_py()] * len(diseases))
            edge_targets_diseases_indices.extend(diseases)

            # Handle linked_targets
            targets = linked_targets.as_py().get('rows', []) if linked_targets and linked_targets.as_py() else []
            edge_sources_indices.extend([molecule_id.as_py()] * len(targets))
            edge_targets_targets_indices.extend(targets)

    
    # Create original indices for edge sources and targets
    original_indices_map = {}

    for i, (edge_source, edge_target_diseases, edge_target_targets) in enumerate(zip(edge_sources_indices, edge_targets_diseases_indices, edge_targets_targets_indices)):
        if edge_source not in original_indices_map:
            original_indices_map[edge_source] = len(original_indices_map)

        if edge_target_diseases not in original_indices_map:
            original_indices_map[edge_target_diseases] = len(original_indices_map)

        if edge_target_targets not in original_indices_map:
            original_indices_map[edge_target_targets] = len(original_indices_map)

    # Create edge_index tensors using original indices
    edge_index = torch.tensor([original_indices_map[edge_sources_indices[i]],
                               original_indices_map[edge_targets_diseases_indices[i]]],
                               dtype=torch.long)

    edge_index_diseases = torch.tensor([original_indices_map[edge_sources_indices[i]],
                                        original_indices_map[edge_targets_diseases_indices[i]]],
                                        dtype=torch.long)

    edge_index_targets = torch.tensor([original_indices_map[edge_sources_indices[i]],
                                       original_indices_map[edge_targets_targets_indices[i]]],
                                       dtype=torch.long)

    # Since nodes are represented by their unique integer indices
    x = torch.arange(max(original_indices_map.values()) + 1, dtype=torch.float).view(-1, 1)

    # Construct and return the Data object
    data = Data(x=x, edge_index=edge_index, edge_index_diseases=edge_index_diseases, edge_index_targets=edge_index_targets)

    return data

def train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1):
    # Ensure the edge_index is a 2D tensor
    assert data.edge_index.dim() == 2, "edge_index should be a 2D tensor"
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1)

    # Split edges into positive train/val/test
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]

    num_val_edges = int(num_edges * val_ratio)
    num_test_edges = int(num_edges * test_ratio)

    test_edge_index = edge_index[:, :num_test_edges]
    val_edge_index = edge_index[:, num_test_edges:num_test_edges+num_val_edges]
    train_edge_index = edge_index[:, num_test_edges+num_val_edges:]

    # Negative sampling
    test_neg_edge_index_diseases = negative_sampling(edge_index=test_edge_index, num_neg_samples=num_test_edges, num_nodes=num_nodes)
    val_neg_edge_index_diseases = negative_sampling(edge_index=val_edge_index, num_neg_samples=num_val_edges, num_nodes=num_nodes)
    train_neg_edge_index_diseases = negative_sampling(edge_index=train_edge_index, num_neg_samples=train_edge_index.size(1), num_nodes=num_nodes)

    test_neg_edge_index_targets = negative_sampling(edge_index=test_edge_index, num_neg_samples=num_test_edges, num_nodes=num_nodes)
    val_neg_edge_index_targets = negative_sampling(edge_index=val_edge_index, num_neg_samples=num_val_edges, num_nodes=num_nodes)
    train_neg_edge_index_targets = negative_sampling(edge_index=train_edge_index, num_neg_samples=train_edge_index.size(1), num_nodes=num_nodes)

    # Ensure no overlap between positive and negative edges (assuming negative_sampling function may not guarantee this)
    for neg_edges, pos_edges in [(test_neg_edge_index_diseases, test_edge_index), 
                                 (val_neg_edge_index_diseases, val_edge_index), 
                                 (train_neg_edge_index_diseases, train_edge_index), 
                                 (test_neg_edge_index_targets, test_edge_index), 
                                 (val_neg_edge_index_targets, val_edge_index), 
                                 (train_neg_edge_index_targets, train_edge_index)]:
        overlap = (neg_edges[0] == pos_edges[0]) & (neg_edges[1] == pos_edges[1])
        assert not overlap.any(), "There's an overlap between positive and negative edges"

    # Update the data object
    data.train_pos_edge_index = train_edge_index
    data.train_neg_edge_index_diseases = train_neg_edge_index_diseases
    data.train_neg_edge_index_targets = train_neg_edge_index_targets
    data.val_pos_edge_index = val_edge_index
    data.val_neg_edge_index_diseases = val_neg_edge_index_diseases
    data.val_neg_edge_index_targets = val_neg_edge_index_targets
    data.test_pos_edge_index = test_edge_index
    data.test_neg_edge_index_diseases = test_neg_edge_index_diseases
    data.test_neg_edge_index_targets = test_neg_edge_index_targets
    
    return data

# GCN Model
class Net(torch.nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)  # Batch normalization
        self.conv2 = GCNConv(32, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)  # Batch normalization

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x  


# Training function
def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Get embeddings for all nodes
    out = model(data)

    # Get embeddings for nodes in train_pos_edge_index and train_neg_edge_index_diseases
    pos_samples = (out[data.train_pos_edge_index[0]] * out[data.train_pos_edge_index[1]]).sum(dim=-1)
    neg_samples_diseases = (out[data.train_neg_edge_index_diseases[0]] * out[data.train_neg_edge_index_diseases[1]]).sum(dim=-1)
    neg_samples_targets = (out[data.train_neg_edge_index_targets[0]] * out[data.train_neg_edge_index_targets[1]]).sum(dim=-1)

    # Concatenate positive and negative samples and labels
    scores_diseases = torch.cat([pos_samples, neg_samples_diseases], dim=0)
    scores_targets = torch.cat([pos_samples, neg_samples_targets], dim=0)
    
    labels = torch.cat([
        torch.ones(pos_samples.shape[0], device=data.x.device),
        torch.zeros(neg_samples_diseases.shape[0] + neg_samples_targets.shape[0], device=data.x.device),
    ])

    labels_diseases = torch.cat([
    torch.ones(pos_samples.shape[0], device=data.x.device),
    torch.zeros(neg_samples_diseases.shape[0], device=data.x.device)
    ])

    loss_diseases = criterion(scores_diseases, labels_diseases)
    loss_targets = criterion(scores_targets, labels[neg_samples_diseases.shape[0]:])
    
    # Combine the losses
    total_loss = loss_diseases + loss_targets
    
    total_loss.backward()
    optimizer.step()
    
    return (loss_diseases.item(), loss_targets.item())

# Main code
if __name__ == "__main__":
    base_path = r"D:\OpenTargets datasets\parquet"
    versions = ["21.04", "21.06", "21.09", "21.11"]
    all_data = []

    for version in versions:
        dataset_path = os.path.join(base_path, version, "molecule")
        
        # Check if the path exists before attempting to load data
        if not os.path.exists(dataset_path):
            print(f"Warning: Directory {dataset_path} does not exist. Skipping this version.")
            continue
        
        data = create_graph_from_parquet(dataset_path)
        # Inspect the edge_index structure
        print(type(data.edge_index))
        print(data.edge_index)
        data.edge_index = data.edge_index.view(2, -1)
            
        data = train_test_split_edges(data)
        if not hasattr(data, 'train_pos_edge_index'):
            print(f"Warning: train_pos_edge_index missing for version {version}. Skipping training for this version.")
            continue
        all_data.append(data)

    # All versions have the same number of features
    num_features = all_data[0].num_features
    
    model = Net(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for data in all_data:
        for epoch in range(100):
            loss_diseases, loss_targets = train(data, model, optimizer, criterion)
            print(f"Epoch: {epoch+1}, Diseases Loss: {loss_diseases:.4f}, Targets Loss: {loss_targets:.4f}")
    
    save_path = input("Enter the path where you want to save the model (including filename): ")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")