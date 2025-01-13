import sys
import os
import torch
import numpy as np
import json
import time
import io
import jsonlines
import json

from IPython import embed
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams
# os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['SLURM_PROCID'])%8)

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



class RecAugDataset(Dataset):
    def __init__(self, input_file, system_prompt, human_prompt_template):
        self.input_file = input_file
        self.system_prompt = system_prompt
        self.human_prompt_template = human_prompt_template
        with open(input_file, 'r') as file:
            sequence_data = json.laod(file)
        self.id_ls = []
        self.squence_dict = {}
        for id, content in sequence_data.items():
            self.id_ls.append(id)
            self.squence_dict[id] = content

        print(f"Dataset load success! Anno Num: {len(self.id_list)}", flush=True)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        try:
           sequence = self.squence_dict[self.id_ls[index]]
           history_list = sequence['history_list']
           potential_items = sequence['potential_items']
           conversation = [{"role":"system", "content":self.system_prompt},
                           {"role":"user", "content":self.human_prompt_template.format(history_list=history_list, potential_items=potential_items)}]
           prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        except Exception as e:
            print(f'{e}, id: {self.id_ls[index]}', flush=True)
        return self.id_ls[index], prompt, history_list, potential_items
        
# def get_prompt(conv):
#     ret = conv.system + conv.sep
#     for role, message in conv.messages:
#         if message:
#             ret += role + ": " + message + conv.sep
#         else:
#             ret += role + ":"
#     return ret


def write_res(exist_dir, id, augmented_sequence, history_list, potential_items):
    with jsonlines.open(exist_dir, 'a') as file:
        for i in range(len(id)):
            data_dict = {}
            data_dict['id'] = id[i]
            data_dict['history_list'] = history_list[i]
            data_dict['potential_items'] = potential_items[i]
            data_dict['augmented_sequence'] = augmented_sequence[i]
            file.write(data_dict)
                
if __name__ == "__main__":
    data_path = "/mnt/hwfile/internvideo/share_data/wuyue/data/SeqRec/toy_file_1_updated.json"
    model_path = '/mnt/hwfile/internvideo/share_data/wuyue/model/LLM_Rec_Qwen2.5_7B_full_sft/'
    exist_dir = ' /mnt/hwfile/internvideo/share_data/wuyue/data/SeqRec/augmented_toy_file_1_updated.jsonl'
    model = LLM(model=model_path, tensor_parallel_size=1)
    dataset = RecAugDataset(data_path, system_prompt, human_prompt_template)
    log_freq = 10
    device = torch.cuda.current_device()
    bs = 512
    n_worker = 32
    shuffle=False
    sampler = None
    dataloader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=None,
            drop_last=False,
            persistent_workers=True if n_worker > 0 else False,
        )
    epoch_time = time.time()
    sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=512)
    for idx, (id, prompt, history_list, potential_items) in enumerate(dataloader):   
        augmented_sequence = model.generate(prompt, sampling_params, use_tqdm=False)
        augmented_sequence = [output.outputs[0].text for output in augmented_sequence]
        write_res(exist_dir, id, augmented_sequence, history_list, potential_items)
        if idx % log_freq == 0:
            print(f"batch idx: {idx}, total: {dataloader.__len__()}, each_{log_freq}_batch_time: {time.time()-epoch_time}", flush=True)
            epoch_time = time.time()