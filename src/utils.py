import torch
import numpy as np
from lora import LinearLayer_LoRA
from datasets import Dataset
from typing import Callable, Literal, Optional, Union


def eigen_preference(num, d):
    x = torch.randint(d, size=[num,])
    x = torch.nn.functional.one_hot(x)
    return x


def gaussian_preference(num, d):
    x = torch.randint(d, size=[num,])
    x = torch.nn.functional.one_hot(x)
    y = torch.randn(size=[num, d])
    return x + y / 9


def uni_preference(num, d):
    x = torch.rand(size=[num])
    return torch.stack([x, 1 - x], dim=1)


def rdit_top_N(dataset, N_user = 5):
    tmp_user_dict = {}
    for i, x in enumerate(dataset['train']['worker']):
        if not x in tmp_user_dict.keys():
            tmp_user_dict[x] = [len(tmp_user_dict), i]
        else:
            tmp_user_dict[x].append(i)
    max_list = [[0, 0] for _ in range(N_user)]
    for x in tmp_user_dict.keys():
        count = len(tmp_user_dict[x]) - 1
        j = 0
        while count <= max_list[j][1]:
            j += 1
            if j == N_user:
                break
        if j < N_user:
            for k in range(N_user - 1, j, -1):
                max_list[k] = max_list[k-1]
            max_list[j] = [x, count]
    
    min_samples = max_list[N_user - 1][1]
    user_dict = {}
    idx = np.zeros(0)
    for i, (x, _) in enumerate(max_list):
        user_dict[x] = i
        idx = np.concatenate([idx, np.random.choice(tmp_user_dict[x][1:], min_samples, replace = False)])

    return user_dict, dataset['train'].select(idx)


def reddit_comp_top_N(dataset, N_worker=5, seed=42):
    """ Filter the dataset to only include the top-N users with the most samples.
    For the reddit TL;DR dataset, we will filter the dataset to only include the top-N users with the most samples on the training set.
    Additionally, we will perform a downsampling to ensure that all top-N users have the same number of samples.
    On the validation set, we will only include the samples from the same top-N on the training set and also perform the same downsampling.
    Args:
        dataset: The reddit TL;DR dataset.
        N_worker: The number of top users to include.
        seed: The random seed for reproducibility.
    """
    # Count the number of samples from each worker on the training/validation set
    worker_count_train = {}
    worker_count_valid = {}
    for i in range(dataset['train'].num_rows):
        worker = dataset['train'][i]['worker']
        if worker not in worker_count_train:
            worker_count_train[worker] = 0
        worker_count_train[worker] += 1
    for i in range(dataset['validation'].num_rows):
        worker = dataset['validation'][i]['worker']
        if worker not in worker_count_valid:
            worker_count_valid[worker] = 0
        worker_count_valid[worker] += 1

    # Sort the users by the number of samples
    sorted_workers_train = sorted(worker_count_train.items(), key=lambda x: x[1], reverse=True)
    print(f"We will filter the dataset to only include the top-{N_worker} users with the most samples on the training set.")
    print(f"[Train] Number of unique workers: {len(sorted_workers_train)}")
    print(f"[Train] Top-5 workers: {sorted_workers_train[:5]}")

    # Filter the dataset to only include the top-N users
    top_N_workers = [worker for worker, _ in sorted_workers_train[:N_worker]]
    filtered_trainset = dataset['train'].filter(lambda x: x['worker'] in top_N_workers)
    filtered_validset = dataset['validation'].filter(lambda x: x['worker'] in top_N_workers)
    print(f"Filtered training samples: {len(filtered_trainset)}")
    print(f"Filtered validation samples: {len(filtered_validset)}")

    # Perform downsampling to ensure that all top-N users have the same number of samples
    min_samples_train = min([count for worker, count in sorted_workers_train[:N_worker]])
    min_samples_valid = min([worker_count_valid[worker] for worker in top_N_workers])
    print(f"Downsampling training set, preserve {min_samples_train} samples for each worker.")
    print(f"Downsampling validation set, preserve {min_samples_valid} samples for each worker.")
    downsampled_trainset = []
    downsampled_validset = []
    for worker in top_N_workers:
        print(f"Worker {worker}")
        worker_samples_train = filtered_trainset.filter(lambda x: x['worker'] == worker)
        worker_samples_valid = filtered_validset.filter(lambda x: x['worker'] == worker)
        downsampled_trainset.extend(worker_samples_train.shuffle(seed=seed).select(range(min_samples_train)))
        downsampled_validset.extend(worker_samples_valid.shuffle(seed=seed).select(range(min_samples_valid)))
    print(f"Downsampled training samples: {len(downsampled_trainset)}")
    print(f"Downsampled validation samples: {len(downsampled_validset)}")

    # Map the worker index to a unique integer index (from 0 to N_worker-1)
    worker_index_map = {worker: i for i, worker in enumerate(top_N_workers)}
    for i in range(len(downsampled_trainset)):
        downsampled_trainset[i]['worker'] = worker_index_map[downsampled_trainset[i]['worker']]
    for i in range(len(downsampled_validset)):
        downsampled_validset[i]['worker'] = worker_index_map[downsampled_validset[i]['worker']]

    # Convert the list of samples to a Hugging Face Dataset
    downsampled_trainset = Dataset.from_list(downsampled_trainset)
    downsampled_validset = Dataset.from_list(downsampled_validset)

    # Construct fine-grained validset for each worker from the downsampled validset
    fine_grained_validset = {}
    for worker_idx in range(N_worker):
        fine_grained_validset[f"labeler{worker_idx}"] = downsampled_validset.filter(lambda x: x['worker'] == worker_idx)

    return downsampled_trainset, downsampled_validset, worker_index_map, fine_grained_validset


def helpsteer_seperate(dataset, args):
    N_worker = 5
    reward_names = [
        'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'
    ]
    def process_entries(batch):
        new_entries = {'chosen': [], 'rejected': [], 'worker': []}
        
        # Pair two consecutive entries
        for i in range(0, len(batch['prompt']), 2):
            prompt1, response1, prompt2, response2 = batch['prompt'][i], batch['response'][i], batch['prompt'][i + 1], batch['response'][i + 1]

            scores1 = [batch[reward_names[j]][i] for j in range(N_worker)]
            scores2 = [batch[reward_names[j]][i + 1] for j in range(N_worker)]

            # Loop through each labeler (1 to 5)
            for labeler in range(N_worker):
                # Compare the scores based on the labeler and determine chosen/reject
                if scores1[labeler] + np.random.random() - 0.5 > scores2[labeler]:
                    chosen = prompt1 + " " + response1
                    reject = prompt2 + " " + response2
                else:
                    chosen = prompt2 + " " + response2
                    reject = prompt1 + " " + response1

                # Append the new entry details to the batch
                new_entries['chosen'].append(chosen)
                new_entries['rejected'].append(reject)
                new_entries['worker'].append(labeler)

        return new_entries

    # Apply the map function
    raw_trainset = dataset['train'].map(
        process_entries, 
        batched=True, 
        batch_size=2, 
        remove_columns=dataset['train'].column_names  # Remove original columns
    )

    raw_testset = dataset['validation'].map(
        process_entries, 
        batched=True, 
        batch_size=2, 
        remove_columns=dataset['validation'].column_names  # Remove original columns
    )

    fine_grained_validset = {}
    for worker_idx in range(N_worker):
        fine_grained_validset[reward_names[worker_idx]] = raw_testset.filter(lambda x: x['worker'] == worker_idx)
    
    return raw_trainset, raw_testset, fine_grained_validset


def reddit_prompt_template(example, response_type: Literal["chosen", "rejected"]):
    """Generate the prompt for the Reddit TL;DR dataset.
    Args:
        example: The example from the Reddit TL;DR dataset.
        response_type: The type of response, either "chosen" or "rejected".
    """
    info = example['info']
    summaries = example['summaries']
    choice = example['choice']
    worker = example['worker']

    subreddit = info["subreddit"]
    title = info["title"]
    post = info["post"]
    assert choice in [0, 1], "The choice must be either 0 or 1."

    if response_type == "chosen":
        summary = summaries[choice]['text']             # chosen response
    else:
        summary = summaries[1 - choice]['text']         # rejected response
    
    prompt = f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR: {summary}"
    
    return prompt
