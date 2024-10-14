import os
import openai
openai.api_key = "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"

import json
output_file_path = "sports_gptoutput_0_shot.jsonl"
model = "gpt-3.5-turbo-0125"
with open("sports_prompt_0_shot.jsonl", "r") as f, open(output_file_path, "w") as out_f:
    print(f"using model {model}")
    for line in f:
        json_line = json.loads(line)  # Parse each line as a JSON object
        prompt = json_line.get('prompt')  # Get the value of the 'prompt' key

        completion = openai.ChatCompletion.create(
          model=model,
          messages=[
            #{"role": "system", "content": "You are an assistant that help me make product recommendation"},
            {"role": "user", "content": prompt}],
          temperature=0,      # Disable randomness
          #top_p=1,
        
        )

        print(completion.choices[0].message)


        message = completion.choices[0].message

        # Write the message to the output JSONL file as a JSON object
        json_line_output = {"completion": message}
        out_f.write(json.dumps(json_line_output) + '\n')
