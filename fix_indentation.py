import json

for file_path in ['kaggle/task1.ipynb', 'kaggle/task2.ipynb', 'kaggle/task3.ipynb']:
    with open(file_path, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # The issue is that the replacement of \\n with \n caused split strings like:
                # print(f"
                # --- Evaluating Video...
                # We need to rejoin lines that were split unnecessarily or fix the quotes.
                # However, the easiest and most robust way is to just regenerate the cells using nbformat natively.
                pass
