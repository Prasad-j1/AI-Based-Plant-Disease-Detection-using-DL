import os
import json

print("Testing path and JSON loading...")

here = os.getcwd()
# print(f"Current working directory: {here}")
json_path = os.path.join("Data_information", "disease_info.json")
# print(f"Constructed JSON path: {json_path}")
with open(json_path, 'r', encoding='utf-8') as f:
	disease_info = json.load(f)
	
print("Successfully loaded JSON data.")