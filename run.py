import argparse
from data_processing import get_data
from rgcn import run_experiment_multiple_times, run_experiment, print_graph_details, plot_analysis, plot_emissions_data, get_resource_consumption_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run RGCN experiments")
    parser.add_argument("--run_once", action="store_true", help="Run the experiment once")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    baseline_data, summarized_data = get_data()
    print_graph_details(baseline_data, summarized_data)
    weight_file = "learned_weights.pth"

    if args.run_once:
        '''RUN EXPERIMENT ONCE'''
        epoch_loss_dict, epoch_accuracy_dict, accuracy_dict, emissions_dict = run_experiment(baseline_data, summarized_data, 2)
        print("Loss over epochs: ", epoch_loss_dict)
        print("Performance over epochs: ", epoch_accuracy_dict)
        print("Test Loss: ", accuracy_dict)
        print("Resource Consumption: ", emissions_dict) 
        plot_analysis(epoch_loss_dict, epoch_accuracy_dict, accuracy_dict)
        csv_file_path = 'emissions.csv'
        emissions_dict = get_resource_consumption_metrics(emissions_dict, csv_file_path)
        plot_emissions_data(emissions_dict)
    else:
        '''RUN EXPERIMENT MULTIPLE TIMES'''
        avg_epoch_loss, avg_epoch_accuracy, avg_accuracy, avg_emissions = run_experiment_multiple_times(baseline_data, summarized_data, weight_file, num_experiments=10)
        print("Loss over epochs: ", avg_epoch_loss)
        print("Performance over epochs: ", avg_epoch_accuracy)
        print("Test Loss: ", avg_accuracy)
        print("Resource Consumption: ", avg_emissions)
        plot_analysis(avg_epoch_loss, avg_epoch_accuracy, avg_accuracy)
        plot_emissions_data(avg_emissions)

