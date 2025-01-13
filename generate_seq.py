import json
from gpt_client_Azure import GPTClient
from tqdm import tqdm
import re

# Initialize GPT client
gpt_client = GPTClient(api_key='test', model_name="gpt-4o", max_retries=3, temperature=0.0, max_tokens=1024, top_p=1.0)


system_prompt = """
You are an intelligent assistant designed to help insert potential items into a user's interaction history sequence.
Given a user's interaction history and a list of potential items, your task is to assess whether it makes sense to insert any of the potential items into the user's history sequence.

Your process is as follows:
1. Analyze the items in the user's interaction history to identify their most recent area of interest.
2. Evaluate whether inserting one or more potential items would reasonably reflect the user's most recent interests or important interest, while maintaining coherence in the sequence.
3. If it is reasonable to insert potential items, generate an updated sequence with the inserted item(s) placed in the appropriate position. 
If it is not reasonable, provide the reason and return a `null` sequence. 
Only generate an updated sequence when you are confident in the decision.

When generating the updated sequence, return it in JSON format, including both `asin` and `timestamp`. 
The timestamp for the inserted items should be determined based on their relationship with the adjacent items, leaning towards the more similar item.
"""

human_prompt_template = """
User Interaction History:
{history_list}

Potential Items:
{potential_items}

Please determine whether any potential items should be inserted into the user's history. If so, generate the updated sequence with appropriate timestamps.

# <FORMAT>
Your output should be in JSON format:
"reason": your reason,
"potential_sequence": [
    {{
        "asin": "<asin>",
        "timestamp": <timestamp>
    }}
] or null
# </FORMAT>

# <Your reason and potential_sequence>
"""


# Initialize the first key
current_key = "0000000000001"

# Example usage within your loop or program
existing_keys = set() 
existing_keys.add(current_key)

def generate_unique_key(current_key):
    # Convert current_key to an integer, increment it, then format it back to a zero-padded string
    next_key_num = int(current_key) + 1
    
    # Return the next key as a 13-character zero-padded string
    return f"{next_key_num:013d}"

# Load your JSON file
with open('/mnt/liuyang/data/toy_file_1_updated.json', 'r') as file:
    data = json.load(file)

index = 0
output_data = {}
# Iterate over each user
for user_id, user_data in tqdm(list(data.items())):  # Convert dictionary items to a list to avoid RuntimeError
    the_last_item = user_data.get('history_list', [])[-1]
    raw_history_list = user_data.get('history_list', [])
    history_list = raw_history_list[:-1] 
    potential_items = user_data.get('potential_items', [])
    
    if not potential_items:  # Skip if no potential items exist for the user
        continue

    # Create the human prompt (now passing all potential items at once)
    human_prompt = human_prompt_template.format(
        history_list=json.dumps(history_list, indent=2),
        potential_items=json.dumps(potential_items, indent=2)  # Pass the whole list of potential items
    )

    # Query GPT with the prompts
    response = gpt_client.query(system_prompt=system_prompt, human_prompt=human_prompt)
    
    # Clean up the response if it starts with code fences
    if response.startswith("```json"):
        response = response[7:-3]
    response = response.replace("\n", "")
    json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
    
    if not json_match:
        json_match = re.search(r'({.*})', response, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            gpt_output = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(response)
            continue
    else:
        print("111")
    
    
    reason = gpt_output.get("reason", "")
    potential_sequence = gpt_output.get("potential_sequence", None)

    # If GPT suggests a potential sequence, add it to the output_data
    if potential_sequence:
        new_key = generate_unique_key(current_key)
        existing_keys.add(new_key)
        current_key = new_key

        # Extend potential_sequence by adding additional information from potential_items
        enhanced_potential_sequence = []
        for seq_item in potential_sequence:
            asin = seq_item['asin']
            timestamp = seq_item['timestamp']
            
            # Find the corresponding item in potential_items
            additional_info = next((item for item in potential_items if item['asin'] == asin), {})
            
            # Combine the `asin`, `timestamp` with the additional information
            enhanced_potential_sequence.append({
                "asin": asin,
                "timestamp": timestamp,
                "title": additional_info.get('title', ''),
                "price": additional_info.get('price', ''),
                "brand": additional_info.get('brand', ''),
                "categories": additional_info.get('categories', ''),
                "description": additional_info.get('description', '')
            })
        enhanced_potential_sequence.sort(key=lambda x: x['timestamp'])
        enhanced_potential_sequence.append(the_last_item)
        output_data[new_key] = {
            "raw_user_id": user_id,
            "history_list": raw_history_list,
            "potential_sequence": enhanced_potential_sequence,
            "reason": reason
        }
    
    index += 1
    if index == 1200:  # Stop after processing 20 users for demonstration
        break

with open('/mnt/liuyang/data/output-1200-1.json', 'w') as file:
    json.dump(output_data, file, indent=2)