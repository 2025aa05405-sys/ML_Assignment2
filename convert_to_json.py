#!/usr/bin/env python3
"""Convert CSV files to JSON format for cloud deployment."""

import csv
import json
import os

# Convert model_results.csv to JSON
if os.path.exists("model_results.csv"):
    with open("model_results.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric strings to floats
            converted_row = {}
            for key, value in row.items():
                try:
                    converted_row[key] = float(value)
                except ValueError:
                    converted_row[key] = value
            rows.append(converted_row)
    
    with open("model/model_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("✓ Converted model_results.csv to model/model_results.json")

# Convert test_data.csv to JSON
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
    
    with open("model/test_data.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("✓ Converted test_data.csv to model/test_data.json")
