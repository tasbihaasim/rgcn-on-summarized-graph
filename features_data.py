import torch_geometric
import torch
data = torch.load('data.pt')

# Step 2: View the data
print(data.edge_index)