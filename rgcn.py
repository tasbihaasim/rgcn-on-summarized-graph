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
node_classes_file = 'node_classes.tsv'

def read_node_classes(file_path):
    """
    Reads a TSV file and categorizes nodes into sets based on column-wise class membership.

    This function processes each row in a TSV file, creating a list of sets where each set contains
    unique nodes belonging to a specific class (column). The function assumes the first row of the
    TSV file is a header.

    Parameters:
        file_path (str): Path to the TSV file containing node class data.

    Returns:
        list[set]: A list of sets, each containing nodes of a particular class.
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

def get_required_mappings(output_file, node_features_file, edge_indices_file):
    '''maps node from original graph to summarized group
    and maps relationship to its corresponding edgeID

    Parameters:
        - output_file (str): Path to the output file containing mappings to summarized nodes.
        - node_features_file (str): Path to the nodeID file containing nodeLabels and their corresponding ID. 
        - edge_indices_file (str): Path to the edgeID file containing edgeType and their corresponding edgeType ID. 


    Returns:
        - Dict[node, summarized_node]: A map containing key = node and value = summarized node that it maps to. 
        - Dict[relationship, edgeType]: A map containing key = relationship and value = numerical representation of edge.
    
    '''
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
    
    return node_to_updatedGroup, relationships_to_edgetype

def get_node_class(node):
    '''
    Gets the node class. 

    Parameters:
        node : node label

    Returns:
        int: numerical value representing node class type
    '''
    
    if node in node_classes[0]:
        return 1
    if node in node_classes[2]:
        return 2
    else:
        return 0

def prepare_data(relationships_to_edgetype, node_to_updatedGroup):
    '''

    Parameters:
        - Dict[node, summarized_node]: A map containing key = node and value = summarized node that it maps to. 
        - Dict[relationship, edgeType]: A map containing key = relationship and value = numerical representation of edge.

    Returns:
        - List[List]: A list of 2 lists. At index[i] of both the list is the two nodes. 
        - Dict[summarized_node, [(node, node_class)]]: A mapping containing summarized nodes and the 
            corresponding nodes from original graph and their class types.
    '''
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
                feature_type_subject = get_node_class(original_subject)
                feature_type_object = get_node_class(original_object)

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
    edge_type = torch.tensor(edge_type)
    return edge_list, edge_type, features

def create_target_matrix(nodes_dict):
    '''
    Gets the target tensor matrix using the features mapping.

    Parameters:
        Dict[summarizednode:[(node, node_class)]] : summarized node and the
        nodes and classtype that are contained within it. 

    Returns:
        tensor matrix: a matrix containing target variable. 

    '''
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

def get_baseline_data(original_file):
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
        feature_type = get_node_class(uri)
        if feature_type is not None:
            y_labels[index][feature_type] = 1

    edge_list = [edge_list1, edge_list2]
    edge_list = torch.tensor(edge_list)
    edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long)
    # assert len(set(edge_list[0])) == tensor_X.shape[0], "Mismatch in lengths of tensor_X and edge_list_subjects"
    X_tensor = y_labels
    return X_tensor, edge_type_tensor, edge_list,  y_labels

def create_sparse_feature_matrix(number_of_nodes):
    # Creating a sparse identity matrix (one-hot encoding)
    # Indices of non-zero elements
    indices = torch.arange(0, number_of_nodes, dtype=torch.long).unsqueeze(0)
    indices = torch.cat((indices, indices), dim=0)

    # Values of non-zero elements
    values = torch.ones(number_of_nodes)

    # Create a sparse tensor
    shape = (number_of_nodes, number_of_nodes)
    feature_matrix_sparse = torch.sparse_coo_tensor(indices, values, shape)

    return feature_matrix_sparse

node_classes = read_node_classes(node_classes_file)    
## get required mapping for summarized graph
node_to_summarized_group, relationships_to_edgetype= get_required_mappings(node_features_file=node_features_file, edge_indices_file=edge_indices_file, output_file=output_file)
## get edge type and edge list and features mapping
result = prepare_data(relationships_to_edgetype, node_to_summarized_group)


def train_rgc_layer(tensor_X, edgeList, edge_type, y_train, num_classes=3, transfer_weights=None, freeze_layers=False, num_epochs=51, lr=0.01):
    in_channels = tensor_X.shape[1]  # Assuming the number of features per node is the second dimension of tensor_X
    out_channels = num_classes
    num_relations = len(set(edge_type))
    hidden_units = 16  # Number of hidden units as per literature

    # Create an instance of RGCNConv without basis decomposition
    rgcn_layer1 = RGCNConv(in_channels, hidden_units, num_relations, num_bases=None, num_blocks=None)
    rgcn_layer2 = RGCNConv(hidden_units, out_channels, num_relations, num_bases=None, num_blocks=None)

    if transfer_weights is not None:
        rgcn_layer1.load_state_dict(transfer_weights['rgcn_layer1'])
        rgcn_layer2.load_state_dict(transfer_weights['rgcn_layer2'])
        if freeze_layers:
            for param in rgcn_layer1.parameters():
                param.requires_grad = False
            for param in rgcn_layer2.parameters():
                param.requires_grad = False

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(rgcn_layer1.parameters()) + list(rgcn_layer2.parameters()), lr=1.0e-2, weight_decay=5.0e-4)  # Adjusted learning rate and weight decay

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


## GET DATA FOR SUMMARIZED MODEL
edge_list_summarized, edge_type_summarized, class_nodes = result # edge list
target_labels_summarized = create_target_matrix(class_nodes) # Y_labels
feature_matrix_summarized = torch.full((len(target_labels_summarized), 1), 1, dtype=torch.float32) # feature matrix


## GET DATA FOR BASELINE MODEL
feature_matrix_baseline, edge_type_baseline, edge_list_baseline, target_labels_baseline = get_baseline_data(original_file)

# Print the number of nodes and edges in the original graph
num_nodes_original = feature_matrix_baseline.shape[0]  # Assuming the number of nodes is the first dimension of features_train
num_edges_original = edge_list_baseline.shape[1]  # Assuming edge_list_train is a 2xN matrix where N is the number of edges

print(f"Original Graph: Number of nodes = {num_nodes_original}")
print(f"Original Graph: Number of edges = {num_edges_original}")

# Print the number of nodes and edges in the summarized graph
num_nodes_summarized = feature_matrix_summarized.shape[0]  # Assuming the number of nodes is the first dimension of feature_matrix_summarized
num_edges_summarized = edge_list_summarized.shape[1]  # Assuming edge_list_summarized is a 2xN matrix where N is the number of edges

print(f"Summarized Graph: Number of nodes = {num_nodes_summarized}")
print(f"Summarized Graph: Number of edges = {num_edges_summarized}")


## first model: summarized graph
learned_weights_summarized = train_rgc_layer(
    feature_matrix_summarized,
    edge_list_summarized,
    edge_type_summarized,
    target_labels_summarized,
    num_classes=3,  
    transfer_weights=None, 
    freeze_layers=False
)

## second model: Transfer weight and Freeze layer is kept True
learned_weights_original = train_rgc_layer(
    feature_matrix_summarized,
    edge_list_summarized,
    edge_type_summarized,
    target_labels_summarized,
    num_classes=3,  
    transfer_weights=learned_weights_summarized,  # weight transfer
    freeze_layers=True   ## freeze layer
)


## third model: Baseline Model 
baseline_model = train_rgc_layer(
    feature_matrix_baseline, 
    edge_type_baseline, 
    edge_list_baseline, 
    target_labels_baseline,
    num_classes=3,  
    transfer_weights=None, 
    freeze_layers=False  
)
