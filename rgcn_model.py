from typing import Optional, Tuple, Union

import torch
from torch import Tensor
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
import torch

def split_data(tensor_X, edgeList, edge_type, ratio, num_nodes):
    # num_nodes = max(torch.max(edgeList[0]).item(), torch.max(edgeList[1]).item()) + 1

    num_train = int(num_nodes * ratio)
    indices = torch.randperm(num_nodes)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    tensor_X_train = tensor_X[train_indices]
    edgeList_train = edgeList[:, train_indices]
    edge_type_train = edge_type[train_indices]

    tensor_X_test = tensor_X[test_indices]
    edgeList_test = edgeList[:, test_indices]
    edge_type_test = edge_type[test_indices]

    return (
        tensor_X_train,
        edgeList_train,
        edge_type_train,
        tensor_X_test,
        edgeList_test,
        edge_type_test,
    )


def get_required_mappings(output_file, node_features_file):
    groups = []
    with open(output_file, "r") as f:
        for line in f:
            group_ID = int(line.strip())
            groups.append(group_ID)

    ## group: updated value
    unique_elements = set(groups)
    updated_value = 0
    group_to_updatedGroup = {}
    updatedGroup_to_group = {}
    for group_value in unique_elements:
        group_to_updatedGroup[group_value] = updated_value
        updatedGroup_to_group[updated_value] = group_value
        updated_value+=1
   
    node_to_updatedGroup = {}
    updatedGroup_to_features = {}
    with open(node_features_file, "r") as f:
        for line in f:
            parts = line.strip().split() 
            node_label = parts[0] # node label
            group_index = int(parts[1]) # node Index
            ug = group_to_updatedGroup[groups[group_index]]
            node_to_updatedGroup[node_label] = ug
            updatedGroup_to_features[ug] = node_label

    relationships_to_edgetype = {}
    with open(edge_indices_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            edge_label = parts[0]
            edge_ID = int(parts[1])
            relationships_to_edgetype[edge_label] =  edge_ID

    return node_to_updatedGroup, updatedGroup_to_features, updatedGroup_to_group, relationships_to_edgetype
    

def get_tensor_X(updatedGroup_to_group, updatedGroup_to_features):
    label_encoder = LabelEncoder()
    matrix = []

    for i in updatedGroup_to_group:
        # Convert string feature to numerical value using label encoding
        numerical_feature = label_encoder.fit_transform([updatedGroup_to_features[i]])[0]
        row = [updatedGroup_to_group[i], numerical_feature]
        matrix.append(row)
    matrix = torch.tensor(matrix)
    return matrix

def get_edge_list(relationships_to_edgetype, node_to_updatedGroup, original_file):
    # Read edge indices from knowledge_graph.ntedge_ID
    ## PREPARE EDGE_LIST AND EDGE_TYPE LIST
    edge_list1 = []
    edge_list2 = []
    edge_type = []
    node_pair_to_relationship = {}
    x = 0
    summarized_node_list = []
    total_number_of_supernodes = 0
    node_list= []
    with open(original_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            # pattern = r'<[^>]+>'
            # matches = re.findall(pattern, data)
            if len(parts)==4:
                subject = node_to_updatedGroup[str(parts[0])] # try abs
                relationship = relationships_to_edgetype[str(parts[1])]
                object = node_to_updatedGroup[str(parts[len(parts)-2])] 
                edge_list1.append(subject)
                edge_list2.append(object)
                edge_type.append(relationship)
    edgeList = [edge_list1, edge_list2]
    edgeList = torch.tensor(edgeList)
    edge_type = torch.tensor(edge_type)
    return (edgeList, edge_type)

def train_rgc_layer(tensor_X_train, edgeList_train, edge_type_train, num_epochs=100, lr=0.01):
    in_channels = 1  # Assuming one feature per node
    out_channels = 3 ## number of classes that can emerge out of RGCN
    num_relations = len(edgeList_train[0])
    num_bases = None
    num_blocks = None

    # Create an instance of RGCNConv
    rgcn_layer = RGCNConv(in_channels, out_channels, num_relations, num_bases=num_bases, num_blocks=num_blocks)
    
    # Loss and optimization setup
    # criterion = torch.nn.CrossEntropyLoss() ## entropy one is a problem
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(rgcn_layer.parameters(), lr=0.01)

    # Example target variable for training (replace with your actual target variable)
    y_train = torch.randint(0, 2, (tensor_X_train.shape[0],), dtype=torch.float)
    print(y_train)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output_train = rgcn_layer(x=tensor_X_train, edge_index=edgeList_train, edge_type=edge_type_train)
        # loss_train = criterion(output_train, y_train)
        loss_train = criterion(input = output_train.view(y_train.size()), target = y_train)
        loss_train.backward() 
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss_train.item()}")
    # Save the learned weights
    learned_weights = rgcn_layer.state_dict()
    torch.save(learned_weights, 'learned_weights.pth')
    return rgcn_layer

def apply_and_evaluate(rgcn_layer, tensor_X_test, edgeList_test, edge_type_test, y_test):
    # Apply the trained model to the test graph
    rgcn_layer.load_state_dict(torch.load('learned_weights.pth'))
    rgcn_layer.eval()
    with torch.no_grad():
        output_test = rgcn_layer(x=tensor_X_test, edge_index=edgeList_test, edge_type=edge_type_test)

    # Evaluate the performance on the test graph
    loss_test = criterion(input=output_test.view(y_test.size()), target=y_test)
    print(f"Test Loss: {loss_test.item()}")

node_features_file = "aifb_stripped.ntnode_ID" ## contains clusters and the number of nodes mapped to the
edge_indices_file = "aifb_stripped.ntedge_ID"
original_file = "aifb_stripped.nt"
output_file = "here.txt"

# STEP 1: DATA PROCESSING
node_to_updatedGroup, updatedGroup_to_features, updatedGroup_to_group, relationships_to_edgetype= get_required_mappings(node_features_file=node_features_file, output_file=output_file)
tensor_X = get_tensor_X(updatedGroup_to_group=updatedGroup_to_group, updatedGroup_to_features=updatedGroup_to_features)
edgeList, edgeType = get_edge_list(relationships_to_edgetype=relationships_to_edgetype, original_file=original_file, node_to_updatedGroup=node_to_updatedGroup)
print("EDGE TYPE", len(edgeType)) # 24370
print("EDGE LIST", len(edgeList))
print("Tensor X", tensor_X.shape)

## Testing edge lists
no_element_lesser = all(element >= 0 for element in edgeList[0])
no_element_greater = all(element <= tensor_X.shape[0] for element in edgeList[0])
print(no_element_greater & no_element_lesser)


train_ratio = 0.8
# STEP 2: split into train and test
tensor_X_train, edgeList_train, edge_type_train, tensor_X_test, edgeList_test, edge_type_test = split_data(
    tensor_X, edgeList, edgeType, train_ratio, total_number_of_supernodes
)
print("tensorx", len(tensor_X_test))
print("edge_index", len(edgeList_test[0]))
print("edge type", len(edge_type_test))
# Assuming you have target labels for the train graph (replace with your actual target variable)
# y_train = torch.randint(0, 2, (tensor_X_train.shape[0],), dtype=torch.long)

# Train RGCN layer
# rgcn_layer = train_rgc_layer(tensor_X_train, edgeList_train, edge_type_train)

# Assuming you have target labels for the test graph (replace with your actual target variable)
# y_test = torch.randint(0, 2, (tensor_X_test.shape[0],), dtype=torch.long)

# Apply and evaluate the trained model on the test graph
# apply_and_evaluate(rgcn_layer, tensor_X_test, edgeList_test, edge_type_test, y_test)


## problems to be solved
# what is target variable
# what is tensor_X