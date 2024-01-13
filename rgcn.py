import torch
from torch_geometric.nn.conv import RGCNConv

import os
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
from data_processing import get_data
import psutil

import matplotlib.pyplot as plt
import pandas as pd
from codecarbon import EmissionsTracker
import uuid

process = psutil.Process()

def evaluate(rgcn_layer, dataset, weights_path=None):
    tensor_X = dataset.x
    edgeList = dataset.edge_index
    edge_type = dataset.edge_attr
    y_test = dataset.y

    # Load weights if provided
    if weights_path is not None:
        print("using transfered weights.. ")
        state_dict = torch.load(weights_path)
        rgcn_layer.load_state_dict(state_dict['rgcn_layer'])

    # Set the model to evaluation mode
    rgcn_layer.eval()

    # Evaluation
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        output_test = rgcn_layer(x=tensor_X, edge_index=edgeList, edge_type=edge_type)
        loss_test = criterion(output_test, y_test)
        print(f"Test Loss: {loss_test.item()}")

    # Additional metrics can be added here (e.g., accuracy, F1-score, etc.)
    # ...

    return loss_test

def calculate_accuracy(outputs, targets):
    # Assuming outputs are logits and targets are one-hot encoded
    predicted_classes = torch.argmax(torch.sigmoid(outputs), dim=1)
    true_classes = torch.argmax(targets, dim=1)
    correct = (predicted_classes == true_classes).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

class TwoLayerRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(TwoLayerRGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x
    
class OneLayerRGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(OneLayerRGCN, self).__init__()
        self.rgcn1 = RGCNConv(in_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn1(x, edge_index, edge_type)
        return x

def train_rgc_layer(data, transfer_weights, freeze_layers, num_epochs=51, lr=0.01, num_layers = 2):
    tensor_X = data.x
    edgeList = data.edge_index
    edge_type = data.edge_attr
    y_train = data.y
    epoch_loss_array = []
    epoch_accuracy_array = []
    num_classes = y_train.shape[1]
    in_channels = tensor_X.shape[1]
    hidden_channels = 16  # Number of hidden units as per literature
    out_channels = num_classes
    num_relations = 45 #torch.unique(edge_type).numel()
    edge_type = edge_type.squeeze()

    if num_layers == 1:
        model = OneLayerRGCN(in_channels, out_channels, num_relations)
    elif num_layers == 2:
        model = TwoLayerRGCN(in_channels, hidden_channels, out_channels, num_relations)
    else:
        raise ValueError("Number of layers must be 1 or 2.")
    
    # Load weights if available
    if transfer_weights is not None:
        model.load_state_dict(transfer_weights['rgcn_layer'])

    # Modify this part to selectively freeze layers
    if freeze_layers:
        for name, param in model.named_parameters():
            if 'rgcn1' in name:  # Example for freezing the first layer
                param.requires_grad = False
                print(f"Freezing {name}")
            else:
                print(f"Training {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        print("No trainable parameters. Skipping training.")
        return model, {'rgcn_layer': model.state_dict()}, epoch_loss_array, epoch_accuracy_array
    learned_weights = {}
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=5.0e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output_train = model(tensor_X, edgeList, edge_type)
        loss_train = criterion(output_train, y_train)
        accuracy_train = calculate_accuracy(output_train, y_train)
        loss_train.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss_train.item()}")
            epoch_loss_array.append(loss_train.item())
            epoch_accuracy_array.append(accuracy_train)

    learned_weights = {'rgcn_layer': model.state_dict()}
    torch.save(learned_weights, 'learned_weights.pth')
    
    return model, learned_weights, epoch_loss_array, epoch_accuracy_array

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

def calculate_resource_utilization():
    # Get the process object for the current process
    process = psutil.Process()

    # Get CPU times (user and system)
    cpu_times = process.cpu_times()
    total_cpu_time = cpu_times.user + cpu_times.system

    # Get memory utilization
    memory_usage = psutil.virtual_memory().percent

    # Return a dictionary with CPU and memory utilization
    return {'cpu_usage': total_cpu_time, 'memory_usage': memory_usage}

def plot_analysis(epoch_loss_dict, epoch_accuracy_dict, accuracy_dict):
    # Convert dictionaries to dataframes for easier plottinimport matplotlib.pyplot as plt

    # Plotting
    epochs = range(1, len(epoch_loss_dict['transfer_model']) + 1)

    # Plotting transfer_model epoch loss in blue
    plt.plot(epochs, epoch_loss_dict['transfer_model'], 'b-', label='Loss Transfer Learning')

    # Plotting transfer_model accuracy in red
    plt.plot(epochs, epoch_accuracy_dict['transfer_model'], 'r-', label='Performance Transfer Learning')

    # Plotting baseline_model epoch loss in yellow
    plt.plot(epochs, epoch_loss_dict['baseline_model'], 'y-', label='Loss Baseline Model')

    # Plotting baseline_model accuracy in green
    plt.plot(epochs, epoch_accuracy_dict['baseline_model'], 'g-', label='Performance Baseline Model')

    # Adding labels and title
    plt.xlabel('Time (epochs)')
    plt.ylabel('Accuracy/Loss')
    plt.title('Time Series Graph of Accuracy and Loss')
    plt.legend()

    # Display the plot
    plt.show()

    # Plot 2: Bar Chart for Test Loss
    plt.figure()
    pd.Series(accuracy_dict).plot(kind='bar', color='skyblue')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.xticks(rotation=45)
    plt.show()

def run_experiment(baseline_data, summarized_data, num_layers):
    # Set the ratios for validation and test sets
    val_ratio = 0.05  # 5% of edges for validation
    test_ratio = 0.1  # 10% of edges for testing

    # Create the transform
    transform = RandomLinkSplit(is_undirected=False, # Set to False since graph is directed
                            num_val=val_ratio, 
                            num_test=test_ratio)

    ## SPLITTING SUMMARIZED GRAPH DATA
    train_data_summarized, val_data_summarized, test_data_summarized = transform(summarized_data)
    ## SPLITTING BASELINE GRAPH DATA
    train_data_baseline, val_data_baseline, test_data_baseline = transform(baseline_data)

    # Initialize dictionaries
    epoch_loss_dict = {'transfer_model': [], 'baseline_model': []}
    epoch_accuracy_dict = {'transfer_model': [], 'baseline_model': []}
    accuracy_dict = {'transfer_model': 0.0, 'baseline_model': 0.0}
    emissions_dict = {'transfer_model': 0.0, 'baseline_model': 0.0}


    # Check if weights file exists, train summarized model if not
    filepath = 'learned_weights.pth'
    if os.path.exists(filepath):
        print("Loading weights from:", filepath)
        transfer_weights = torch.load(filepath)
    else:
        print("Training the summarized model...")
        summarized_model, transfer_weights, epoch_loss_array_summarized, epoch_accuracy_array_summarized = train_rgc_layer(
            train_data_summarized,
            transfer_weights=None,
            freeze_layers=False,
            num_layers=num_layers
        )
        torch.save(transfer_weights, filepath)

    print("Training the transfer model...")
    tracker_transfer = EmissionsTracker()
    tracker_transfer.start()

    transfer_model, weights_transfer_model, epoch_loss_array_transfer, epoch_accuracy_array_transfer = train_rgc_layer(
        train_data_summarized,
        transfer_weights=transfer_weights,
        freeze_layers=True, 
        num_layers=num_layers
    )

    # Stop tracker and fetch data
    run_id_transfer = tracker_transfer.run_id
    tracker_transfer.stop()

    emissions_dict['transfer_model'] = run_id_transfer
    accuracy_dict['transfer_model'] = float(evaluate(transfer_model, test_data_summarized, weights_path=None))


    # Train the baseline model
    print("Training the baseline model...")
    tracker_baseline = EmissionsTracker()
    tracker_baseline.start()

    baseline_model, weights_baseline_model, epoch_loss_array_baseline, epoch_accuracy_array_baseline = train_rgc_layer(
        train_data_baseline,
        transfer_weights=None,
        freeze_layers=False, 
        num_layers=num_layers
    )

    # Stop tracker and fetch data
    run_id_baseline = tracker_baseline.run_id
    tracker_baseline.stop()

    emissions_dict['baseline_model'] = run_id_baseline
    accuracy_dict['baseline_model'] = float(evaluate(baseline_model, test_data_baseline, weights_path=None))

    epoch_loss_dict['transfer_model'] = epoch_loss_array_transfer
    epoch_loss_dict['baseline_model'] = epoch_loss_array_baseline
    epoch_accuracy_dict['transfer_model'] = epoch_accuracy_array_transfer
    epoch_accuracy_dict['baseline_model'] = epoch_accuracy_array_baseline

    return epoch_loss_dict, epoch_accuracy_dict, accuracy_dict, emissions_dict

def get_resource_consumption_metrics(run_id_dict, csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Initialize an empty dictionary to store the results
    resource_consumption_metrics = {}

    # Iterate over each run_id in the dictionary
    for model, run_id in run_id_dict.items():
        # Ensure that run_id is a string
        run_id_str = str(run_id) if isinstance(run_id, uuid.UUID) else run_id

        # Find the row in the DataFrame that matches this run_id
        run_data = df[df['run_id'] == run_id_str]

        if not run_data.empty:
            metrics = {
                'emissions': run_data['emissions'].iloc[0],
                'emissions_rate': run_data['emissions_rate'].iloc[0],
                'cpu_energy': run_data['cpu_energy'].iloc[0],
                'ram_energy': run_data['ram_energy'].iloc[0],
                'energy_consumed': run_data['energy_consumed'].iloc[0]
            }
            resource_consumption_metrics[model] = metrics
        else:
            resource_consumption_metrics[model] = {
                'emissions': None,
                'emissions_rate': None,
                'cpu_energy': None,
                'ram_energy': None,
                'energy_consumed': None
            }

    return resource_consumption_metrics

def plot_emissions_data(data_dict):
    categories = list(data_dict['transfer_model'].keys())
    transfer_values = list(data_dict['transfer_model'].values())
    baseline_values = list(data_dict['baseline_model'].values())

    bar_width = 0.35
    index = range(len(categories))

    plt.figure(figsize=(12, 6))

    plt.bar(index, transfer_values, bar_width, label='Transfer Model')
    plt.bar([i + bar_width for i in index], baseline_values, bar_width, label='Baseline Model')

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Comparison of Transfer Model and Baseline Model Metrics')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_experiment_multiple_times(baseline_data, summarized_data, weight_file, num_experiments=10):
    # Initialize empty lists to store results
    all_epoch_loss = []
    all_epoch_accuracy = []
    all_accuracy = []
    all_emissions = []

    # Run the experiment multiple times
    for _ in range(num_experiments):
        if _ %2 == 0:
            remove_file(weight_file)
        print("Experiment no. ", _)
        # Run the experiment once
        epoch_loss_dict, epoch_accuracy_dict, accuracy_dict, emissions_dict = run_experiment(baseline_data, summarized_data, 2)
        csv_file_path = 'emissions.csv'
        emissions_dict = get_resource_consumption_metrics(emissions_dict, csv_file_path)
        # Append results to the lists
        all_epoch_loss.append(epoch_loss_dict)
        all_epoch_accuracy.append(epoch_accuracy_dict)
        all_accuracy.append(accuracy_dict)
        all_emissions.append(emissions_dict)

    # Calculate average values
    avg_epoch_loss_dict = {
        'transfer_model': np.mean([exp['transfer_model'] for exp in all_epoch_loss], axis=0).tolist(),
        'baseline_model': np.mean([exp['baseline_model'] for exp in all_epoch_loss], axis=0).tolist()
    }

    avg_epoch_accuracy_dict = {
        'transfer_model': np.mean([exp['transfer_model'] for exp in all_epoch_accuracy], axis=0).tolist(),
        'baseline_model': np.mean([exp['baseline_model'] for exp in all_epoch_accuracy], axis=0).tolist()
    }

    avg_accuracy_dict = {
        'transfer_model': np.mean([exp['transfer_model'] for exp in all_accuracy]),
        'baseline_model': np.mean([exp['baseline_model'] for exp in all_accuracy])
    }

    avg_emissions_dict = {
        'transfer_model': {
            key: np.mean([exp['transfer_model'][key] for exp in all_emissions]) for key in emissions_dict['transfer_model']
        },
        'baseline_model': {
            key: np.mean([exp['baseline_model'][key] for exp in all_emissions]) for key in emissions_dict['baseline_model']
        }
    }

    return avg_epoch_loss_dict, avg_epoch_accuracy_dict, avg_accuracy_dict, avg_emissions_dict

def remove_file(file_name):
    # Construct the full path to the file
    file_path = os.path.join(os.getcwd(), file_name)
    # Check if the file exists before trying to remove it
    if os.path.exists(file_path):
        # Remove the file
        os.remove(file_path)
        print(f"{file_name} removed successfully.")
    else:
        print(f"{file_name} does not exist.")










