

import json

# Specify the input JSONL file
input_file = 'first_last_item.jsonl'

# Open and read the file line by line
with open(input_file, 'r') as f:
    for line in f:
        # Parse the JSON object from each line
        data = json.loads(line)
        
        # Access item1 and item2 from the JSON object
        item1 = data.get("item1")
        item2 = data.get("item2")
        
        # Process or print the values


        template = f"seed item is {item1}, recommended item is {item2}; "
        print(template)

