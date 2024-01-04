from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from rdflib import Graph
from torch.nn import Parameter
from torch.nn import Parameter as Param
import tensorflow as tf
import numpy as np
import torch_geometric.backend
from sklearn.preprocessing import LabelEncoder
import torch_geometric.typing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)

from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch_geometric.utils.sparse import index2ptr

import os
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from torch_geometric.utils import train_test_split_edges
import csv
import rdflib
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
import numpy as np
from data_processing import get_data

def evaluate(rgcn_layer, dataset, weights_path=None):
    tensor_X = dataset.x
    edgeList = dataset.edge_index
    edge_type = dataset.edge_attr
    y_test = dataset.y

    # Load weights if provided
    if weights_path is not None:
        print("using transfered weights.. ")
        state_dict = torch.load(weights_path)
        rgcn_layer.load_state_dict(state_dict['rgcn_layer'])

    # Set the model to evaluation mode
    rgcn_layer.eval()

    # Evaluation
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        output_test = rgcn_layer(x=tensor_X, edge_index=edgeList, edge_type=edge_type)
        loss_test = criterion(output_test, y_test)
        print(f"Test Loss: {loss_test.item()}")

    # Additional metrics can be added here (e.g., accuracy, F1-score, etc.)
    # ...

    return loss_test

class TwoLayerRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(TwoLayerRGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x

def train_rgc_layer(data, transfer_weights, freeze_layers, num_epochs=51, lr=0.01):
    tensor_X = data.x
    edgeList = data.edge_index
    edge_type = data.edge_attr
    y_train = data.y

    num_classes = y_train.shape[1]
    in_channels = tensor_X.shape[1]
    hidden_channels = 16  # Number of hidden units as per literature
    out_channels = num_classes
    num_relations = torch.unique(edge_type).numel()
    edge_type = edge_type.squeeze()

    model = TwoLayerRGCN(in_channels, hidden_channels, out_channels, num_relations)

    # Load weights if available
    if transfer_weights is not None:
        model.load_state_dict(transfer_weights['rgcn_layer'])

    # Modify this part to selectively freeze layers
    if freeze_layers:
        for name, param in model.named_parameters():
            if 'rgcn1' in name:  # Example for freezing the first layer
                param.requires_grad = False
                print(f"Freezing {name}")
            else:
                print(f"Training {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        print("No trainable parameters. Skipping training.")
        return model, {'rgcn_layer': model.state_dict()}

    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=5.0e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output_train = model(tensor_X, edgeList, edge_type)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss_train.item()}")

    learned_weights = {'rgcn_layer': model.state_dict()}
    torch.save(learned_weights, 'learned_weights.pth')
    return model, learned_weights

def print_graph_details(baseline_data, summarized_data):
    # Print details for baseline data
    num_nodes_original = baseline_data.x.shape[0]
    num_edges_original = baseline_data.edge_index.shape[1]
    num_relation_types_original = torch.unique(baseline_data.edge_attr)
    num_classes_original = baseline_data.y.shape[1]
    num_features_original = baseline_data.x.shape[1]

    print(f"Baseline Data: Number of nodes = {num_nodes_original}")
    print(f"Baseline Data: Number of edges = {num_edges_original}")
    print(f"Baseline Data: Number of relation types = {num_relation_types_original.numel()}")
    print(f"Baseline Data: Number of classes = {num_classes_original}")
    print(f"Baseline Data: Number of features = {num_features_original}")

    # Print details for summarized data
    num_nodes_summarized = summarized_data.x.shape[0]
    num_edges_summarized = summarized_data.edge_index.shape[1]
    num_relation_types_summarized = torch.unique(summarized_data.edge_attr)
    num_classes_summarized = summarized_data.y.shape[1]
    num_features_summarized = summarized_data.x.shape[1]

    print(f"Summarized Data: Number of nodes = {num_nodes_summarized}")
    print(f"Summarized Data: Number of edges = {num_edges_summarized}")
    print(f"Summarized Data: Number of relation types = {num_relation_types_summarized.numel()}")
    print(f"Summarized Data: Number of classes = {num_classes_summarized}")
    print(f"Summarized Data: Number of features = {num_features_summarized}")

baseline_data, summarized_data = get_data()

print_graph_details(baseline_data, summarized_data)


# Set the ratios for validation and test sets
val_ratio = 0.05  # 5% of edges for validation
test_ratio = 0.1  # 10% of edges for testing

# Create the transform
transform = RandomLinkSplit(is_undirected=False, # Set to False since graph is directed
                            num_val=val_ratio, 
                            num_test=test_ratio)

## SPLITTING SUMMARIZED GRAPH DATA
train_data_summarized, val_data_summarized, test_data_summarized = transform(summarized_data)
## SPLITTING BASELINE GRAPH DATA
train_data_baseline, val_data_baseline, test_data_baseline = transform(baseline_data)


# Function to train the first model or load weights if they already exist
def train_or_load_first_model():
    filepath = 'learned_weights.pth'
    if os.path.exists(filepath):
        print("Loading weights from:", filepath)
        return torch.load(filepath)
    else:
        print("Training the first model...")
        first_model, learned_weights = train_rgc_layer(
            train_data_summarized,
            transfer_weights=None,
            freeze_layers=False
        )
        evaluate(first_model, test_data_summarized, weights_path='learned_weights.pth')
        evaluate(first_model, test_data_summarized, weights_path=None)
        torch.save(learned_weights, filepath)
        return learned_weights

# Train the first model or load weights
transfer_weights = train_or_load_first_model()

# Train the second model with transfer learning and layer freezing
print("TRAINING THE WEIGHT TRANSFER MODEL...")
second_model_rgcn_layer, weights_second_model = train_rgc_layer(
    train_data_summarized,
    transfer_weights=transfer_weights,  # Use transferred weights
    freeze_layers=True  # Freeze layer
)


print("EVALUATING THE SUMMARIZED GRAPH ON SECOND MODEL")
# apply_and_evaluate(second_model_rgcn_layer, test_data_summarized.x, test_data_summarized.edge_index, test_data_summarized.edge_attr, test_data_summarized.y, num_classes=3)
evaluate(second_model_rgcn_layer, test_data_summarized, weights_path='learned_weights.pth')
evaluate(second_model_rgcn_layer, test_data_summarized, weights_path=None)
# third model: Baseline Model 
print("TRAINING THE BASELINE MODEL ...")
baseline_model_rgcn_layer, weights_baseline_model = train_rgc_layer(
    train_data_baseline,
    transfer_weights=None, 
    freeze_layers=False  
)

print("EVALUATING THE BASELINE GRAPH DATA ON BASELINE MODEL")
# apply_and_evaluate(baseline_model_rgcn_layer, test_data_baseline.x, test_data_baseline.edge_index, test_data_baseline.edge_attr, test_data_baseline.y, num_classes=3)
evaluate(baseline_model_rgcn_layer, test_data_baseline, weights_path=None)