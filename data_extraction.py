from torch_geometric.datasets import Entities
from torch_geometric.transforms import ToSparseTensor

# Specify the root directory where the dataset should be saved
root = './data'

# Specify the name of the dataset ("AIFB")
dataset_name = 'AIFB'

# Define a transform to convert the data to a sparse tensor (you can customize or omit this)
transform = ToSparseTensor()

# Create an instance of the Entities class to load the AIFB dataset
aifb_dataset = Entities(root=root, name=dataset_name, hetero=True, transform=transform)

data = aifb_dataset.load
# Print some information about the dataset
print(f'Dataset: {data.info()}:')
# print(f'Number of graphs: {len(aifb_dataset)}')
# print(f'Number of classes: {aifb_dataset.num_classes}')
# print(f'Number of features: {aifb_dataset.num_node_features}')

# # Accessing a specific graph in the dataset (e.g., the first graph)
# graph_idx = 0
# data = aifb_dataset[graph_idx]
# print('\nGraph Information:')
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(aifb_dataset.process())
# print(f'Node features shape: {data.size}')
# print(f'Edge indices shape: {data.edge_index.}')
# print(f'Graph label: {data.y}')

# Additional processing or analysis can be performed based on your specific task
