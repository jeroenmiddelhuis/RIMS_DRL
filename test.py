import json
        
with open('./example/RL_integration/input.json', 'r') as f:
    input_data = json.load(f)

resources = sorted(list(input_data["resource"].keys()))
activities = sorted(list(input_data["processing_time"].keys()))
print(resources)
print(activities)