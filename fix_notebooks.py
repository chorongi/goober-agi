import json
import os

for nb_file in ['kaggle/task1.ipynb', 'kaggle/task2.ipynb', 'kaggle/task3.ipynb']:
    with open(nb_file, 'r') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        new_source = []
        for line in cell['source']:
            # Replace literal "\\n" at the end of the line with a real "\n"
            if line.endswith("\\n"):
                new_source.append(line[:-2] + "\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        
    with open(nb_file, 'w') as f:
        json.dump(nb, f, indent=1)
        
