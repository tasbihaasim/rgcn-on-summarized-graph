from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
import tensorflow as tf
import numpy as np
import torch_geometric.backend
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
import torch


# Step 1: Read your graph data from files
node_features_file = "dummy_graph.ntnode_ID" ## contains clusters and the number of nodes mapped to the
edge_indices_file = "dummy_graph.ntedge_ID"
original_file = "dummy_graph.nt"
output_file = "here.txt"

# Read group mapping file. 
# At index = node_ID, we have the group_value. 
groups = []
with open(output_file, "r") as f:
    for line in f:
        group_value = int(line.strip())
        groups.append(group_value)

node_to_index = {}
node_to_weight = {} #
number_of_nodes = 0  # number of clusters in summarized graph + singletons
total_groups = len(groups)

with open(node_features_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        node_id = parts[0]
        group_index = int(parts[1])
        node_to_index[node_id] = group_index
        if group_index < total_groups:
            group_value = groups[group_index]
            node_to_weight[node_id] = group_value
        else:
            # Handles singletons
            node_to_weight[node_id] = group_index  # whatever name is coming this way from node_features_file
        number_of_nodes += 1 ## need to do this in case skipSingleton was true, then here.txt wont be the same size

#print(node_to_weight)
## PREPARE FEATURE MATRIX X. we do not have feature for now for summarized version, so it can be any tensor.
tensor_X = torch.tensor([1.0] * number_of_nodes).view(-1, 1)
print(tensor_X)

## HASHMAP OF RELATIONSHIPS
## just maps relationship name to their corresponding numerical value
relationships_to_edgetype = {}
with open(edge_indices_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        edge_label = parts[0]
        edge_ID = int(parts[1])
        relationships_to_edgetype[edge_label] =  edge_ID


# Read edge indices from knowledge_graph.ntedge_ID
## Aggregate relationships
edge_list1 = []
edge_list2 = []
edge_type = []
node_pair_to_relationship = {}
with open(original_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        subject = node_to_index[str(parts[0])]
        relationship = relationships_to_edgetype[str(parts[1])]
        object = node_to_index[str(parts[2])]
        node_pair = (subject, object)
        if node_pair in node_pair_to_relationship:
            node_pair_to_relationship[node_pair] = node_pair_to_relationship[node_pair] + relationship
        else:
            node_pair_to_relationship[node_pair] = relationship

## PREPARE EDGE_LIST AND EDGE_TYPE LIST
for key in node_pair_to_relationship.keys():
    edge_list1.append(key[0])
    edge_list2.append(key[1])
    edge_type.append(node_pair_to_relationship[key])

edgeList = [edge_list1, edge_list2]
edgeList = torch.tensor(edgeList)
print("EDGE LIST", edgeList)
edge_type = torch.tensor(edge_type)
print(edge_type)


# # Step 3: Create an instance of RGCNConv
in_channels = 1  # Assuming one feature per node
out_channels = 1
num_relations = len(edgeList[0])
num_bases = None
num_blocks = None

## Step 4: RGCN layer initialized
rgcn_layer = RGCNConv(in_channels, out_channels, num_relations, num_bases=num_bases, num_blocks=num_blocks)
print(rgcn_layer)


# # Call the forward function
output = rgcn_layer(x=tensor_X, edge_index=edgeList, edge_type=edge_type)

# # Step 5: Test the model
print("Output shape:", output.shape)


'''LINK PREDICTION'''
# # Step 5: Train the model (assuming you have a target variable y)
# # Loss and optimization setup
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rgcn_layer.parameters(), lr=0.01)

# # Example target variable (should be replace with your actual target variable)
y = torch.rand(tensor_X.shape[0], 1) 

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = rgcn_layer(x=tensor_X, edge_index=edgeList, edge_type=edge_type)
    print(output) ## the dimension of output should be the same as Y
    loss = criterion(output, y) ## matches link predictions with actual links in Y
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0: ## prints epoch at every 10th instance
        print(f"Epoch {epoch}, Loss: {loss.item()}")


'''ENTITY CLASSIFICATION'''

# # Assuming you have a graph with node features tensor_X and node labels y
# # You might need a separate dataset for training and testing

# # Loss and optimization setup
# criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
# optimizer = torch.optim.Adam(rgcn_layer.parameters(), lr=0.01)

# # Example node labels (replace with your actual node labels)
# y = torch.randint(0, num_classes, (tensor_X.shape[0],))

# # Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     output = rgcn_layer(x=tensor_X, edge_index=edge_index, edge_type=edge_type)
    
#     # Assuming output is a tensor with shape (num_nodes, num_classes)
#     loss = criterion(output, y)
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")

# # After training, you can use the trained model for inference on new data
# # For example, if you have a separate test set tensor_X_test, edge_index_test, edge_type_test:
# rgcn_layer.eval()  # Set the model to evaluation mode
# output_test = rgcn_layer(x=tensor_X_test, edge_index=edge_index_test, edge_type=edge_type_test)
# predicted_labels = torch.argmax(output_test, dim=1)

# # Now you can evaluate the performance of the model on the test set
# # You might use metrics like accuracy, precision, recall, F1-score, etc.
