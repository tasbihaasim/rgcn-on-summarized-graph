
import torch
from torch_geometric.nn import RGCNConv

# Step 1: Generate dummy data
num_nodes = 200
num_relations = 100
num_features = 16
num_edges = 500

# Generate random node features
x = torch.randn(num_nodes, num_features)
print("PRINTING X, SHAPE:", x.shape)
print(x)
# Generate random edge indices
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
print("EDGE INDEX, SHAPE: ", edge_index.shape)
print(edge_index)

# Generate random edge types
edge_type = torch.randint(0, num_relations, (num_edges,), dtype=torch.long)
print("EDGE TYPE, SHAPE:", edge_type.shape)
print(edge_type)

# Generate random weights for each edge
weights = torch.randn(num_edges)

# Step 2: Create an instance of RGCNConv
in_channels = num_features
out_channels = 32
num_bases = None
num_blocks = None

rgcn_layer = RGCNConv(in_channels, out_channels, num_relations, num_bases=num_bases, num_blocks=num_blocks)

# Step 3: Pass the data through the RGCN layer
output = rgcn_layer(x, edge_index, edge_type)

# Step 4: Test the model
print("Input node features shape:", x.shape)
print("Output node features shape:", output.shape)

