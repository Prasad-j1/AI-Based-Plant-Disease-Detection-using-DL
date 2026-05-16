import json
import os

here = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(here, "Data_information", "disease_info.json")
with open(json_path, 'r', encoding='utf-8') as f:
	disease_info = json.load(f)

print(disease_info["Apple___healthy"]['prevention'])