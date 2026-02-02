import json

with open('data/ai-todv2/annotations_v1/aitod_train.json', 'r') as f:
    data = json.load(f)

print(data.keys())
for i in data['categories']:
    print(i)
