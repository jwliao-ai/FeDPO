import datasets
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
from typing import Dict, List, Union, Tuple


def extract_anthropic_prompt(prompt_and_response):

    """Extract the anthropic prompt from a prompt and response pair."""
    
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):

    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children
                                if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(
    split,
    silent=False,
    cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """

    print(f'Loading SE dataset ({split} split) from local cache...')
    dataset = datasets.load_dataset('./datasets/stack-and-exchange', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses  # a list of answers
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])  # the response giving the highest score
        data[prompt]['truncation_mode'] = 'keep_start'

    return data


def get_shp(
    split: str,
    silent: bool = False,
    cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    
    """Load the Stanford Human Preferences dataset from Local and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """

    print(f'Loading SHP dataset ({split} split) from Local cache...')
    dataset = datasets.load_dataset('./datasets/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)
        data[prompt]['truncation_mode'] = 'keep_start'

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'],
                                         key=lambda x: data[prompt]['scores']
                                         [data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(
    split: str,
    silent: bool = False,
    cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """

    print(f'Loading HH dataset ({split} split) from local cache...')
    dataset = datasets.load_dataset('./datasets/hh-rlhf',
                                    split=split,
                                    cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['truncation_mode'] = 'keep_end'

    return data


def get_dataset(name_list: "list[str]",
                split: str,
                silent: bool = False,
                cache_dir: str = None,
                client_num_in_total: int = 1,
                data_evenly_distributed: bool = True,
                use_small_dataset: bool = True):
    """Load the given dataset by name and split it into 'client_num_in_total + 1 (for global test)' parts.
       Supported by default are 'shp', 'hh', and 'se'."""
    
    if split == 'test':
        test_dataset = {}
        for name_place, name in enumerate(name_list):
            if name == 'shp':
                data = get_shp(split, silent=silent, cache_dir=cache_dir)
            elif name == 'hh':
                data = get_hh(split, silent=silent, cache_dir=cache_dir)
            elif name == 'se':
                data = get_se(split, silent=silent, cache_dir=cache_dir)
            else:
                raise ValueError(f"Unknown dataset '{name}'")
            
            assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target', 'truncation_mode'}, \
                f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
            
            test_dataset.update(data)

        return test_dataset
    
    else:
        split_local_datasets = []
        global_dataset = {}

        for name_place, name in enumerate(name_list):
            if name == 'shp':
                data = get_shp(split, silent=silent, cache_dir=cache_dir)
            elif name == 'hh':
                data = get_hh(split, silent=silent, cache_dir=cache_dir)
            elif name == 'se':
                data = get_se(split, silent=silent, cache_dir=cache_dir)
            else:
                raise ValueError(f"Unknown dataset '{name}'")

            assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target', 'truncation_mode'}, \
                f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

            total_keys = len(data)
            keys = list(data.keys())
            
            if use_small_dataset:
                total_keys = total_keys // 10
                keys = random.sample(keys, total_keys)
                data = {key: data[key] for key in keys}

            if data_evenly_distributed == True:
                step_size = int(total_keys * 0.8) // client_num_in_total
                idx = list(range(step_size, total_keys, step_size))
                idx.append(int(total_keys * 0.8))
            else:
                idx = random.sample(range(0, int(total_keys * 0.8)), client_num_in_total - 1)
                idx.sort()
                idx.append(int(total_keys * 0.8))

            for i in range(client_num_in_total + 1):

                if i != client_num_in_total:
                    part_keys = keys[idx[i-1] if i != 0 else 0:idx[i]]
                    if name_place == 0:
                        split_local_datasets.append({key: data[key] for key in part_keys})
                    else:
                        split_local_datasets[i].update({key: data[key] for key in part_keys})
                else:
                    end = total_keys
                    part_keys = keys[idx[i-1]:end]
                    if name_place == 0:
                        global_dataset = {key: data[key] for key in part_keys}
                    else:
                        global_dataset.update({key: data[key] for key in part_keys})
            del keys
            
        return split_local_datasets, global_dataset
    