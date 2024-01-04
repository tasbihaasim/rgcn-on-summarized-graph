
import torch
from rdflib import Graph
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges
import csv
from collections import defaultdict
from torch_geometric.data import Data
import numpy as np
import re

## open files
node_features_file = "aifb_stripped.ntnode_ID" ## contains clusters and the number of nodes mapped to the
edge_indices_file = "aifb_stripped.ntedge_ID"
original_file = "aifb_stripped.nt"
output_file = "here.txt"
node_classes_file = 'testSet.tsv'

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
            edge_label = parts[0][1:-1]
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
    if node in node_classes[1]:
        return 2
    if node in node_classes[2]:
        return 3
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
    missed_entries = 0
    graph = Graph()
    edge_list1 = []
    edge_list2 = []
    edge_type = []
    features={}
    with open(original_file, "r") as f:
        for line in f:
            graph.parse(data=line, format="nt")
            parts = line.strip().split()      
            if len(parts)>2:
                original_subject = parts[0][1:-1]
                original_object = parts[2][1:-1]
                if (original_subject not in node_to_updatedGroup) or (original_object not in node_to_updatedGroup):
                    pass
                else:
                    summarized_subject = node_to_updatedGroup[original_subject] # try abs
                    summarized_object = node_to_updatedGroup[original_object] 
                    relationship = relationships_to_edgetype[str(parts[1][1:-1])]
                    
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
            else:
                missed_entries+=1
                
    edge_list = [edge_list1, edge_list2]
    edge_list = torch.tensor(edge_list)
    edge_type = torch.tensor(edge_type)
    return edge_list, edge_type, features

def create_target_matrix(nodes_dict, num_classes):
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
        
        class_count = [0] * num_classes
        total_weight = 0
        row = []
        for node, class_type in nodes_dict[group_id]:
            class_count[class_type] = class_count[class_type]+1
            total_weight+=1
        for i in class_count:
            prob_val = i/total_weight
            row.append(prob_val)
        matrix.append(row)
    return torch.tensor(matrix)

def get_baseline_data(original_file):
    # Load RDF data from NT file
    graph = Graph()
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
            graph.parse(data=line, format="nt")
            parts = line.strip().split()      
            if len(parts)>2:
                subj = parts[0][1:-1]
                obj = parts[2][1:-1]
                # print(parts[1])
                pred = str(parts[1][1:-1])
                pred = relationships_to_edgetype[pred]
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
    y_labels = torch.zeros((len(node_dict), 4))  # Assuming two features for each node
    # X_tensor = torch.zeros(len(node_dict), 4)

    for uri, index in node_dict.items():
        feature_type = get_node_class(uri)
        if feature_type is not None:
            y_labels[index][feature_type] = 1

    edge_list = [edge_list1, edge_list2]
    edge_list = torch.tensor(edge_list)
    edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long)
    # assert len(set(edge_list[0])) == tensor_X.shape[0], "Mismatch in lengths of tensor_X and edge_list_subjects"
    X_tensor = torch.full((len(y_labels), 1), 1, dtype=torch.float32)
    return X_tensor, edge_type_tensor, edge_list,  y_labels

node_classes = read_node_classes(node_classes_file)   
num_classes = len(node_classes) + 1 
## get required mapping for summarized graph
node_to_summarized_group, relationships_to_edgetype= get_required_mappings(node_features_file=node_features_file, edge_indices_file=edge_indices_file, output_file=output_file)
## get edge type and edge list and features mapping
result = prepare_data(relationships_to_edgetype, node_to_summarized_group)

## GET DATA FOR SUMMARIZED MODEL
edge_list_summarized, edge_type_summarized, class_nodes = result # edge list and edgetype
target_labels_summarized = create_target_matrix(class_nodes, num_classes) # Y_labels
feature_matrix_summarized = torch.full((len(target_labels_summarized), 1), 1, dtype=torch.float32) # feature matrix

## GET DATA FOR BASELINE MODEL
feature_matrix_baseline, edge_type_baseline, edge_list_baseline, target_labels_baseline = get_baseline_data(original_file)


# Prepare data using Data object
summarized_data = Data(x=feature_matrix_summarized, edge_index=edge_list_summarized, 
                       edge_attr=edge_type_summarized, y=target_labels_summarized)

baseline_data = Data(x=feature_matrix_baseline, edge_index=edge_list_baseline, 
                     edge_attr=edge_type_baseline, y=target_labels_baseline)


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


# print_graph_details(baseline_data, summarized_data)

def get_data():
    return baseline_data, summarized_data