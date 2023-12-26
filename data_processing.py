import csv
import re
import torch
import numpy as np


# Replace these with the actual file paths
csv_file_path = "aifb_stripped.ntnode_ID"
output_file_path = "here.txt"
node_id_file = "aifb_stripped.ntnode_ID"
relationship_file = "aifb_stripped.ntedge_ID"
nt_tuple_file = "aifb_stripped.nt"

## # Function to read the node_ID file and create a mapping of node labels to node IDs

def read_node_id_file(node_id_file):
    node_to_ID_mapping = {}
    with open(node_id_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            node_label = parts[0]
            node_id = int(parts[1])
            node_to_ID_mapping[node_label] = node_id
    #print(node_to_ID_mapping)
    return node_to_ID_mapping

# Function to read the relationship file and create a mapping of relationship labels to relationship IDs
def read_relationship_file(relationship_file):
    relationship_mapping = {}
    with open(relationship_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                relationship_label, relationship_id = parts
                try:
                    relationship_mapping[relationship_label] = int(relationship_id)
                except ValueError:
                    print(f"Error parsing relationship ID for relationship label: {relationship_label}")
    #print(relationship_mapping)
    return relationship_mapping

# Function to process the nt tuple file and create the edge list and edge type list
def process_nt_tuple_file(nt_tuple_file):
    edge_list = []
    edge_type_list =[]
    with open(nt_tuple_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts)==4:
                subject = str(parts[0]) 
                relationship = str(parts[1])
                object = str(parts[2])
                node1_id = node_id_mapping[subject]
                node2_id = node_id_mapping[object]
                relationship_id = relationship_mapping[relationship]
                if node1_id is not None and node2_id is not None and relationship_id is not None:
                    edge_list.append([node1_id, node2_id])
                    edge_type_list.append(relationship_id)
        return edge_list, edge_type_list




# Initialize an empty array
node_array = []

# Read the values from here.txt
with open(output_file_path, mode="r", newline='') as here_file:
    here_values = [int(value.strip()) for value in here_file]

# Read the CSV file and populate the array
with open(csv_file_path, mode="r", newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=' ')
    for index, row in enumerate(reader):
        # Assuming the first value is the node label and the second value is the node ID
        node_label = row[0]
        node_id = int(row[1])

        # Ensure that the index is within bounds
        if 0 <= index < len(here_values):
            here_value = here_values[index]

            # Append None values to the array until reaching the current node_id
            while len(node_array) <= node_id:
                node_array.append(None)

            # Populate the array with the node_label and here_value at the corresponding node_id
            node_array[node_id] = [node_label, here_value]

# Convert the node_array to a PyTorch tensor
#tensor_X = torch.tensor(node_array)

# Print the resulting tensor


# Read node_ID file
node_id_mapping = read_node_id_file(node_id_file)
#print(node_id_mapping)

# Read relationship file
relationship_mapping = read_relationship_file(relationship_file)
#print(relationship_mapping)

# Process nt tuple file
edge_list, edge_type_list = process_nt_tuple_file(nt_tuple_file)

# Print the resulting edge list and edge type list
print("Edge List:", len(edge_list))
print("Edge Type List:", len(edge_type_list))
#print(node_array)
