import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

def find_json_files(base_path, dataset_size):
    # Use absolute path
    base_path = os.path.abspath(base_path)
    
    # Search pattern for .pth files
    pattern = os.path.join(base_path, f"*_GNN_passing_*_*_{dataset_size}.pth")
    model_files = glob.glob(pattern)
    print(f"Found {len(model_files)} model files:")
    
    # Group by passing number
    passing_groups = {}
    for model_file in model_files:
        print(f"\nProcessing file: {model_file}")
        model_dir = os.path.dirname(model_file)
        
        # Extract passing number with updated regex
        match = re.search(r'GNN_passing_(\d+)_', model_file)
        if match:
            passing_num = int(match.group(1))
            print(f"Found passing number: {passing_num}")
            
            json_file = os.path.join(model_file, "metrics.json")
            print(f"Looking for JSON at: {json_file}")
            
            if os.path.exists(json_file):
                if passing_num not in passing_groups:
                    passing_groups[passing_num] = []
                passing_groups[passing_num].append(json_file)
                print(f"Added metrics file for passing {passing_num}")
        else:
            print(f"No passing number found in filename: {os.path.basename(model_file)}")
    
    if not passing_groups:
        print("No valid metrics files found")
        
    return passing_groups


def load_metrics_data(passing_groups):
    metrics_data = {}
    
    for passing_num, json_files in passing_groups.items():
        metrics_data[passing_num] = {
            'GNN': {
                'RMSE_error': [],
                'MSE_error': [],
                'L2_error': [],
                'Relative_RMSE': []
            },
            'FC': {
                'RMSE_error': [],
                'MSE_error': [],
                'L2_error': [],
                'Relative_RMSE': []
            }
        }
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Store GNN metrics
                for metric in metrics_data[passing_num]['GNN'].keys():
                    if metric in data['GNN']:
                        metrics_data[passing_num]['GNN'][metric].append(data['GNN'][metric])
                
                # Store FC metrics
                for metric in metrics_data[passing_num]['FC'].keys():
                    if metric in data['FC']:
                        metrics_data[passing_num]['FC'][metric].append(data['FC'][metric])
    
    return metrics_data

def create_plots(metrics_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # Line plot
    passing_numbers = sorted(metrics_data.keys())
    rmse_values = [metrics_data[p]['GNN']['RMSE_error'][0]['mean'] for p in passing_numbers]
    rmse_stds = [metrics_data[p]['GNN']['RMSE_error'][0]['std'] for p in passing_numbers]
    rel_rmse_values = [metrics_data[p]['GNN']['Relative_RMSE'][0]['mean'] for p in passing_numbers]
    rel_rmse_stds = [metrics_data[p]['GNN']['Relative_RMSE'][0]['std'] for p in passing_numbers]
    
    ax1.errorbar(passing_numbers, rmse_values, yerr=rmse_stds, fmt='o-', label='RMSE')
    ax1.errorbar(passing_numbers, rel_rmse_values, yerr=rel_rmse_stds, fmt='s-', label='Relative RMSE')
    ax1.set_xlabel('Passing Number')
    ax1.set_ylabel('Error [m]')
    ax1.set_title('Error Progression with Passing Number')
    ax1.grid(True)
    ax1.legend()
    
    # Histogram
    labels = ['FC'] + [f'GNN_{p}' for p in passing_numbers]
    x = np.arange(len(labels))
    width = 0.35
    
    # FC values (same for all passes, use first one)
    fc_rmse = metrics_data[passing_numbers[0]]['FC']['RMSE_error'][0]['mean']
    fc_rmse_std = metrics_data[passing_numbers[0]]['FC']['RMSE_error'][0]['std']
    fc_rel_rmse = metrics_data[passing_numbers[0]]['FC']['Relative_RMSE'][0]['mean']
    fc_rel_rmse_std = metrics_data[passing_numbers[0]]['FC']['Relative_RMSE'][0]['std']
    
    rmse_values = [fc_rmse]
    rmse_stds = [fc_rmse_std]
    rel_rmse_values = [fc_rel_rmse]
    rel_rmse_stds = [fc_rel_rmse_std]
    
    # GNN values for each passing
    for p in passing_numbers:
        rmse_values.append(metrics_data[p]['GNN']['RMSE_error'][0]['mean'])
        rmse_stds.append(metrics_data[p]['GNN']['RMSE_error'][0]['std'])
        rel_rmse_values.append(metrics_data[p]['GNN']['Relative_RMSE'][0]['mean'])
        rel_rmse_stds.append(metrics_data[p]['GNN']['Relative_RMSE'][0]['std'])
    
    ax2.bar(x - width/2, rmse_values, width, yerr=rmse_stds, label='RMSE')
    ax2.bar(x + width/2, rel_rmse_values, width, yerr=rel_rmse_stds, label='Relative RMSE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('Error Comparison')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_rmse_line_plot(metrics_data):
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    
    passing_numbers = sorted(metrics_data.keys())
    rmse_values = [metrics_data[p]['GNN']['RMSE_error'][0]['mean'] for p in passing_numbers]
    rmse_stds = [metrics_data[p]['GNN']['RMSE_error'][0]['std'] for p in passing_numbers]
    
    ax1.errorbar(passing_numbers, rmse_values, yerr=rmse_stds, fmt='o-', label='RMSE')
    ax1.set_xlabel('Passing Number')
    ax1.set_ylabel('RMSE [m]')
    ax1.set_title('RMSE Progression with Passing Number')
    ax1.grid(True)
    ax1.legend()
    return fig1

def create_rel_rmse_line_plot(metrics_data):
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    
    passing_numbers = sorted(metrics_data.keys())
    rel_rmse_values = [metrics_data[p]['GNN']['Relative_RMSE'][0]['mean'] for p in passing_numbers]
    rel_rmse_stds = [metrics_data[p]['GNN']['Relative_RMSE'][0]['std'] for p in passing_numbers]
    
    ax2.errorbar(passing_numbers, rel_rmse_values, yerr=rel_rmse_stds, fmt='s-', label='Relative RMSE')
    ax2.set_xlabel('Passing Number')
    ax2.set_ylabel('Relative RMSE')
    ax2.set_title('Relative RMSE Progression with Passing Number')
    ax2.grid(True)
    ax2.legend()
    return fig2

def create_rmse_histogram(metrics_data):
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    
    passing_numbers = sorted(metrics_data.keys())
    labels = ['FC'] + [f'GNN_{p}' for p in passing_numbers]
    x = np.arange(len(labels))
    width = 0.8
    
    fc_rmse = metrics_data[passing_numbers[0]]['FC']['RMSE_error'][0]['mean']
    fc_rmse_std = metrics_data[passing_numbers[0]]['FC']['RMSE_error'][0]['std']
    
    rmse_values = [fc_rmse] + [metrics_data[p]['GNN']['RMSE_error'][0]['mean'] for p in passing_numbers]
    rmse_stds = [fc_rmse_std] + [metrics_data[p]['GNN']['RMSE_error'][0]['std'] for p in passing_numbers]
    
    ax2.bar(x, rmse_values, width, yerr=rmse_stds)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('RMSE Comparison')
    ax2.set_ylabel('RMSE [m]')
    ax2.grid(True, alpha=0.3)
    return fig2

def create_rel_rmse_histogram(metrics_data):
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    
    passing_numbers = sorted(metrics_data.keys())
    labels = ['FC'] + [f'GNN_{p}' for p in passing_numbers]
    x = np.arange(len(labels))
    width = 0.8
    
    fc_rel_rmse = metrics_data[passing_numbers[0]]['FC']['Relative_RMSE'][0]['mean']
    fc_rel_rmse_std = metrics_data[passing_numbers[0]]['FC']['Relative_RMSE'][0]['std']
    
    rel_rmse_values = [fc_rel_rmse] + [metrics_data[p]['GNN']['Relative_RMSE'][0]['mean'] for p in passing_numbers]
    rel_rmse_stds = [fc_rel_rmse_std] + [metrics_data[p]['GNN']['Relative_RMSE'][0]['std'] for p in passing_numbers]
    
    ax3.bar(x, rel_rmse_values, width, yerr=rel_rmse_stds)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title('Relative RMSE Comparison')
    ax3.set_ylabel('Relative RMSE')
    ax3.grid(True, alpha=0.3)
    return fig3

def save_plots(metrics_data, dataset_size):
    fig1 = create_rmse_line_plot(metrics_data)
    fig2 = create_rel_rmse_line_plot(metrics_data)
    fig3 = create_rmse_histogram(metrics_data)
    fig4 = create_rel_rmse_histogram(metrics_data)
    
    if not os.path.exists('metrics_plots'):
        os.makedirs('metrics_plots')
    if not os.path.exists('metrics_plots/' + dataset_size):
        os.makedirs('metrics_plots/' + dataset_size)

    fig1.savefig(f'metrics_plots/{dataset_size}/rmse_line_plot.png')
    fig2.savefig(f'metrics_plots/{dataset_size}/rel_rmse_line_plot.png')
    fig3.savefig(f'metrics_plots/{dataset_size}/rmse_histogram.png')
    fig4.savefig(f'metrics_plots/{dataset_size}/rel_rmse_histogram.png')

if __name__ == "__main__":
    base_path = "metrics_data"
    dataset_size = "2k"
    passing_groups = find_json_files(base_path, dataset_size)
    metrics_data = load_metrics_data(passing_groups)
    
    #save plots
    save_plots(metrics_data, dataset_size)