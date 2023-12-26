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
import rdflib

## open files
node_features_file = "aifb_stripped.ntnode_ID" ## contains clusters and the number of nodes mapped to the
edge_indices_file = "aifb_stripped.ntedge_ID"
original_file = "aifb_stripped.nt"
output_file = "here.txt"
features_file = 'trainingSet.tsv'

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
    if node in train_features[0]:
        return 1
    if node in train_features[2]:
        return 2
    else:
        return 0

def in_set(element, array_of_sets):
    for s in array_of_sets:
        if element in s:
            return True
    return False

def prepare_data(relationships_to_edgetype, node_to_updatedGroup):
    graph = Graph()
    edge_list1 = []
    edge_list2 = []
    edge_type = []
    features={}
    with open(original_file, "r") as f:
        for line in f:
            graph.parse(data=line, format="nt")
            parts = line.strip().split()           
            if len(parts)==4:
                original_subject = parts[0][1:-1]
                original_object = parts[2][1:-1]
                summarized_subject = node_to_updatedGroup[original_subject] # try abs
                summarized_object = node_to_updatedGroup[original_object] 
                relationship = relationships_to_edgetype[str(parts[1])]

                edge_list1.append(summarized_subject)
                edge_type.append(relationship)
                edge_list2.append(summarized_object)
                feature_type_subject = get_feature_type(original_subject)
                feature_type_object = get_feature_type(original_object)

                if summarized_subject in features:
                    features[summarized_subject].append((original_subject, feature_type_subject))
                if summarized_subject not in features:
                    features[summarized_subject] = [(original_subject, feature_type_object)]

                if summarized_object in features:
                    features[summarized_object].append((original_object, feature_type_subject))
                if summarized_object not in features:
                    features[summarized_object] = [(original_object, feature_type_object)]
                
                    
                    

    edge_list = [edge_list1, edge_list2]
    edge_list = torch.tensor(edge_list)
    # print(edge_list1_train)
    # if (len(set(edge_list_train))) == len(features_train):
    #     print("True")
    edge_type = torch.tensor(edge_type)
    return edge_list, edge_type, features

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

def construct_baseline_model(original_file):
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
    y_labels = torch.zeros((len(node_dict), 3))  # Assuming two features for each node
    # X_tensor = torch.zeros(len(node_dict), 4)

    for uri, index in node_dict.items():
        feature_type = get_feature_type(uri)
        if feature_type is not None:
            y_labels[index][feature_type] = 1
            

    edge_list = [edge_list1, edge_list2]
    edge_list = torch.tensor(edge_list)
    edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long)
    # assert len(set(edge_list[0])) == tensor_X.shape[0], "Mismatch in lengths of tensor_X and edge_list_subjects"
    X_tensor = y_labels
    return X_tensor, edge_list, edge_type_tensor, y_labels

def train_rgc_layer(tensor_X, edgeList, edge_type, y_train, num_classes=3, transfer_weights=None, freeze_layers=False, num_epochs=51, lr=0.01):
    in_channels = tensor_X.shape[1]  # Assuming the number of features per node is the second dimension of tensor_X
    print(tensor_X.shape)
    out_channels = num_classes
    num_relations = len(set(edge_type))

    # Create an instance of RGCNConv
    rgcn_layer1 = RGCNConv(in_channels, 16, num_relations, num_bases=None, num_blocks=None)
    rgcn_layer2 = RGCNConv(16, out_channels, num_relations, num_bases=None, num_blocks=None)

    if transfer_weights is not None:
        rgcn_layer1.load_state_dict(transfer_weights['rgcn_layer1'])
        rgcn_layer2.load_state_dict(transfer_weights['rgcn_layer2'])
        if freeze_layers:
            for param in rgcn_layer1.parameters():
                param.requires_grad = False
            for param in rgcn_layer2.parameters():
                param.requires_grad = False

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(rgcn_layer1.parameters()) + list(rgcn_layer2.parameters()), lr=lr, weight_decay=5e-4)


    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x_intermediate = rgcn_layer1(x=tensor_X, edge_index=edgeList, edge_type=edge_type)
        output_train = rgcn_layer2(x=x_intermediate, edge_index=edgeList, edge_type=edge_type)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss_train.item()}")

    learned_weights = {'rgcn_layer1': rgcn_layer1.state_dict(), 'rgcn_layer2': rgcn_layer2.state_dict()}
    torch.save(learned_weights, 'learned_weights.pth')
    return learned_weights

def apply_and_evaluate(rgcn_layer, tensor_X, edgeList, edge_type, y_variable):
    # Apply the trained model to the test graph
    rgcn_layer.load_state_dict(torch.load('learned_weights.pth'))
    rgcn_layer.eval()
    with torch.no_grad():
        output_test = rgcn_layer(x=tensor_X, edge_index=edgeList, edge_type=edge_type)

    # Evaluate the performance on the test graph
    loss_test = criterion(input=output_test.view(y_test.size()), target=y_variable)
    print(f"Test Loss: {loss_test.item()}")


# Example usage
# train_rgc_layer(tensor_X, edgeList, edge_type, num_classes=4)
# To transfer weights and freeze layers:
# transferred_weights = train_rgc_layer(...)
# train_rgc_layer(tensor_X, edgeList, edge_type, num_classes=4, transfer_weights=transferred_weights, freeze_layers=True)


train_features = read_tsv(features_file)

node_to_updatedGroup, updatedGroup_to_group, relationships_to_edgetype= get_required_mappings(node_features_file=node_features_file, output_file=output_file)

tensor_X_original, edge_index_original, edge_type_original, y_labels_original = construct_baseline_model(original_file)

result = prepare_data(relationships_to_edgetype, node_to_updatedGroup)
edge_list_summarized, edge_type_summarized, features_summarized = result

feature_matrix_summarized = create_feature_matrix(features_summarized)
y_labels_summarized = feature_matrix_summarized


train_rgc_layer(tensor_X_original, edge_index_original, edge_type_original, y_labels_original)
# train_rgc_layer(feature_matrix_summarized, edge_list_summarized, edge_type_summarized, y_labels_summarized, 3, freeze_layers=True)


## TODO:



# Yes, in the context of node classification in a graph using a model like an R-GCN 
# (Relational Graph Convolutional Network), the concepts of feature matrices (X), 
# target labels (Y), and loss calculation are directly related to how well 
# the model predicts the class of each node. Here's how these concepts tie into node classification: