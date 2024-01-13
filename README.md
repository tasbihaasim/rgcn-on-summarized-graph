# Energy Consumption Analysis of RGCN Training on Summarized Graphs

## For Windows Users

### Running the k-Bisimulation Execution File

To generate the summarized representation of the graph:

```bash
./full_bisimulation.exe run_timed mappingbased-objects_lang=en.ttl

./full_bisimulation.exe run_k_bisimulation_store_partition mappingbased-objects_lang=en.ttl --output=here.txt 

./full_bisimulation.exe run_k_bisimulation_store_partition dummy_file.nt --k=3 --output=output.txt --support=5
```
Replace "dummy_file" with your actual file name, and adjust the values of 'k' and 'support' parameters if needed.

## Setting Up the Python Environment

Install the required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Running the Python Code

After activating the environment, run the following command to run the experiment for n number of times:

```bash
python run.py
```

If you want to run the experiment only once, then use the following command:

```bash
python run.py run_once
```

This Readme provides instructions for executing the k-Bisimulation process on a Windows system and setting up the required Python environment to analyze the energy consumption during the training of Relational Graph Convolutional Networks (RGCN) on summarized graphs. Follow the steps carefully to replicate and assess the energy consumption analysis.






