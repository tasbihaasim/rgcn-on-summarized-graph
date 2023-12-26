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
import torch
import csv

## open files
node_features_file = "aifb_stripped.ntnode_ID" ## contains clusters and the number of nodes mapped to the
edge_indices_file = "aifb_stripped.ntedge_ID"
original_file = "aifb_stripped.nt"
output_file = "here.txt"
test_file_path = 'testSet.tsv'
train_file_path = 'trainingSet.tsv'

def read_tsv(file_path):
    """ 
    """
    column_sets = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)  # Assuming the first row is the header

        # Initialize sets for each column
        column_sets = [set() for _ in range(len(header))]

        # Read each row and add entries to respective sets
        for row in reader:
            for i, value in enumerate(row):
                column_sets[i].add(value)

    return column_sets

def get_required_mappings(output_file, node_features_file):
    '''maps node from original graph to summarized group
    and maps relationship to its corresponding number'''
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
            node_label = parts[0][1:-1] # node label
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
    
    return node_to_updatedGroup, updatedGroup_to_group, relationships_to_edgetype

def get_feature_type(node):
    if node in test_features[0] or node in train_features[0]:
        return 1
    if node in test_features[2] or node in train_features[2]:
        return 2
    else:
        return 0

def in_set(element, array_of_sets):
    for s in array_of_sets:
        if element in s:
            return True
    return False

def split_data(relationships_to_edgetype, node_to_updatedGroup):
    graph = Graph()
    edge_list1_train = []
    edge_list2_train = []
    edge_list1_test = []
    edge_list2_test= []
    edge_type_test = []
    edge_type_train = []
    features_test = {}
    features_train = {}
    array_of_nodes = set()
    with open(original_file, "r") as f:
        for line in f:
            graph.parse(data=line, format="nt")
            parts = line.strip().split()           
            if len(parts)>2:
                original_subject = parts[0][1:-1]
                original_object = parts[2][1:-1]
                if original_subject in node_to_updatedGroup:
                    summarized_subject = node_to_updatedGroup[original_subject] # try abs
                if original_object in node_to_updatedGroup:
                    summarized_object = node_to_updatedGroup[original_object] 

                relationship = relationships_to_edgetype[str(parts[1])]
                
                if in_set(original_subject,test_features): ## subject is in test distribution
                    edge_list1_test.append(summarized_subject)
                    edge_type_test.append(relationship)
                    edge_list2_test.append(summarized_object)
                    feature_type = get_feature_type(original_subject)
                    if summarized_subject in features_test:
                        features_test[summarized_subject].append((original_subject, feature_type))
                    if summarized_subject not in features_test:
                        features_test[summarized_subject] = [(original_subject, feature_type)]

                if in_set(original_object,test_features):
                    feature_type = get_feature_type(original_object)
                    if summarized_object in features_test:
                        features_test[summarized_object].append((original_object, feature_type))
                    if summarized_object not in features_test:
                        features_test[summarized_object] = [(original_object, feature_type)]
                    
                if in_set(original_subject,train_features):
                    edge_list1_train.append(summarized_subject)
                    edge_type_train.append(relationship)
                    edge_list2_train.append(summarized_object)
                    feature_type = get_feature_type(original_subject)
                    if summarized_subject in features_train:
                        features_train[summarized_subject].append((original_subject, feature_type))
                    if summarized_subject not in features_train:
                        features_train[summarized_subject] = [(original_subject, feature_type)]

                if in_set(original_object,train_features):
                    feature_type = get_feature_type(original_object)
                    if summarized_object in features_train:
                        features_train[summarized_object].append((original_subject, feature_type))
                    if summarized_object not in features_train:
                        features_train[summarized_object] = [(original_subject, feature_type)]

    edge_list_train = [edge_list1_train, edge_list2_train]
    edge_list_train = torch.tensor(edge_list_train)
    # print(edge_list1_train)
    # if (len(set(edge_list_train))) == len(features_train):
    #     print("True")
    edge_list_test = torch.tensor([edge_list1_test, edge_list2_test])
    edge_type_test = torch.tensor(edge_type_test)
    edge_type_train = torch.tensor(edge_type_train)
    return edge_list_test, edge_type_test, features_test, edge_list_train, edge_type_train,  features_train

def create_feature_matrix(nodes_dict):
    ## iterate over updated Group = total number of columns
    matrix=[]
    count = 0
    for group_id in nodes_dict:
        
        class_count = [0,0,0]
        total_weight = 0
        row = []
        for node, class_type in nodes_dict[group_id]:
            class_count[class_type] = class_count[class_type]+1
            total_weight+=1
        for i in class_count:
            prob_val = i/total_weight
            row.append(prob_val)
        # sprint(row)
        matrix.append(row)
    return torch.tensor(matrix)

def validity_check(feature_matrix, edge_type, edge_list):
    if len(edge_type) == len(edge_list[0]):
        print("here")
        if len(set(edge_list[0])) == len(feature_matrix):
            return True
    else:
        return False


import rdflib

def construct_baseline_model(file_name):
    # Load RDF data from NT file
    g = Graph()

    # Process RDF data to create tensors for RGCN
    node_dict = {}
    edge_list = []
    edge_type_dict = {}  # Dictionary for edge types
    edge_type_list = []
    node_index = 0
    edge_type_index = 0
    edge_list1 = []
    edge_list2 = []
    with open(original_file, "r") as f:
        for line in f:
            g.parse(data=line, format="nt")
            parts = line.strip().split()           
            if len(parts)>2:
                subj = parts[0][1:-1]
                obj = parts[2][1:-1]
                # print(parts[1])
                pred = relationships_to_edgetype[str(parts[1])]
                # Node processing
                if subj not in node_dict:
                    node_dict[subj] = node_index
                    node_index += 1
                if obj not in node_dict:
                    node_dict[obj] = node_index
                    node_index += 1
                edge_type_list.append(pred)
                edge_list1.append(node_dict[subj])
                edge_list2.append(node_dict[obj])

    # Tensor creation
    tensor_X = torch.zeros((len(node_dict), 4))  # Assuming two features for each node
    y_labels = torch.zeros(len(node_dict))

    for uri, index in node_dict.items():
        feature_type = get_feature_type(uri)
        if feature_type is not None:
            tensor_X[index][feature_type] = 1
            y_labels[index] = feature_type

    edge_list = [edge_list1, edge_list2]
    edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long)
    edge_list2 = edge_list2 + edge_list1
    # assert len(set(edge_list[0])) == tensor_X.shape[0], "Mismatch in lengths of tensor_X and edge_list_subjects"

    return tensor_X, edge_list, edge_type_tensor, y_labels

test_features = read_tsv(test_file_path)
train_features = read_tsv(train_file_path)

node_to_updatedGroup, updatedGroup_to_group, relationships_to_edgetype= get_required_mappings(node_features_file=node_features_file, output_file=output_file)

tensor_X, edge_index, edge_type, y_labels = construct_baseline_model(original_file)

result = split_data(relationships_to_edgetype, node_to_updatedGroup)
edge_list_test, edge_type_test, features_test, edge_list_train, edge_type_train, features_train = result
    
feature_matrix_test = create_feature_matrix(features_test)
feature_matrix_train = create_feature_matrix(features_train)

print(validity_check(feature_matrix_test, edge_type_test, edge_list_test))
print(validity_check(feature_matrix_train, edge_type_train, edge_list_train))


# def contruct_y_variable(tensor_X):
#     for i in range(len(tensor_X)):


def train_rgc_layer(tensor_X, edgeList, edge_type, num_epochs=100, lr=0.01):
    in_channels = 3  # Assuming one feature per node
    out_channels = in_channels ## number of classes that can emerge out of RGCN
    num_relations = len(edge_type_train)
    num_bases = None
    num_blocks = None

    # Create an instance of RGCNConv
    rgcn_layer = RGCNConv(in_channels, out_channels, num_relations, num_bases=num_bases, num_blocks=num_blocks)
    # Loss and optimization setup
    criterion = torch.nn.CrossEntropyLoss() ## entropy one is a problem
    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(rgcn_layer.parameters(), lr=lr, weight_decay=5e-4)

    # Example target variable for training (replace with your actual target variable)
    y_train = tensor_X

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output_train = rgcn_layer(x=tensor_X, edge_index=edgeList, edge_type=edge_type)
        loss_train = criterion(output_train, y_train)
        # loss_train = criterion(input = output_train.view(y_train.size()), target = y_train)
        loss_train.backward() 
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss_train.item()}")
    # Save the learned weights
    learned_weights = rgcn_layer.state_dict()
    torch.save(learned_weights, 'learned_weights.pth')
    return rgcn_layer


train_rgc_layer(feature_matrix_train, edge_list_train, edge_type_train)

## TODO:
# For the summarized graph, Y would have the same structure as X, 
# representing the weighted labels for each summary node. This results in:
## construct the original graph tomorrow
## we need the original graph for baseline mode
## Original graph: one hot encoding for X
## Y would be the same as summarized one

# Yes, in the context of node classification in a graph using a model like an R-GCN 
# (Relational Graph Convolutional Network), the concepts of feature matrices (X), 
# target labels (Y), and loss calculation are directly related to how well 
# the model predicts the class of each node. Here's how these concepts tie into node classification: