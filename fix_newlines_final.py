import json

files = ['kaggle/task1.ipynb', 'kaggle/task2.ipynb', 'kaggle/task3.ipynb']

for file_path in files:
    with open(file_path, 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if 'source' in cell:
            new_source = []
            for line in cell['source']:
                if line.endswith('\\n'):
                    new_source.append(line[:-2] + '\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)

print("Done fixing literal newlines.")
