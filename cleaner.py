import os
import json

def modify_versor_in_json():
    directory = "npy_GNN_hf"
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "info.json":
                file_path = os.path.join(root, file)
                
                # Read JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Modify versor to keep only first 3 values
                if 'force_info' in data and 'versor' in data['force_info']:
                    data['force_info']['versor'] = data['force_info']['versor'][:3]
                
                # Write back
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                print(f"Modified {file_path}")

if __name__ == "__main__":
    modify_versor_in_json()