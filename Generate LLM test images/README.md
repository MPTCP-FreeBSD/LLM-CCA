## Script: `llm_graphs test.ipynb`

This script is designed to visualize the test results of LLM-CCA. It processes the data generated from running LLM-CCA tests and creates graphical representations to help analyze the model's performance.

### Usage Instructions
To use this script, you need to modify the following parameters:
- **`log_dir`**: Change this to the path of the folder where the JSON files generated from the test runs are stored. This ensures that the script correctly locates the test results for visualization.
- **`csv_file_path`**: Update this to the path of the CSV file used to generate the PKL files for testing. This ensures that the data aligns correctly when visualizing the performance.

### Example
Make sure to replace the placeholders with the correct paths before running the script:
```python
log_dir = 'path/to/your/json/folder'
csv_file_path = 'path/to/your/csv/file'
