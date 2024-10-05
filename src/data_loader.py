from accelerate import PartialState
from datasets import load_dataset
from trl.extras.dataset_formatting import conversations_formatting_function

from utils import reddit_comp_top_N, reddit_prompt_template, helpsteer_seperate, helpsteer_prompt_template

def get_dataset(args, config, tokenizer):
    flag_summarize = 'summarize_from_feedback' in args.dataset_name
    flag_help = 'HelpSteer' in args.dataset_name
    
    if flag_summarize:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_subset)
        raw_trainset, raw_testset, worker_dict, fine_grained_validset = reddit_comp_top_N(raw_datasets, args.num_labelers)
    elif flag_help:
        raw_datasets = load_dataset(args.dataset_name)
        raw_trainset, raw_testset, fine_grained_validset = helpsteer_seperate(raw_datasets, args)

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "labeler_index": examples["worker"],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen, padding="longest", truncation=True, max_length=config.max_length)
            tokenized_rejected = tokenizer(rejected, padding="longest", truncation=True, max_length=config.max_length)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the chosen/rejected columns are in the OpenAI messages format.
        if flag_summarize:
            if args.apply_chat_template:
                chosen_fn = conversations_formatting_function(tokenizer, "chosen")
                rejected_fn = conversations_formatting_function(tokenizer, "rejected")
                raw_datasets = raw_datasets.map(
                    lambda x: {"chosen": chosen_fn(x), "rejected": rejected_fn(x)},
                    num_proc=config.dataset_num_proc
                )
                # Tokenize inputs
                raw_datasets = raw_datasets.map(
                    preprocess_function,
                    batched=True,
                    num_proc=config.dataset_num_proc,
                )
                # Filter out examples that are too long
                raw_datasets = raw_datasets.filter(
                    lambda x: len(x["input_ids_chosen"]) <= config.max_length
                    and len(x["input_ids_rejected"]) <= config.max_length,
                    num_proc=config.dataset_num_proc,
                )
            else:
                raw_trainset = raw_trainset.map(
                    lambda x: {"chosen": reddit_prompt_template(x, "chosen"),
                            "rejected": reddit_prompt_template(x, "rejected")},
                    num_proc=config.dataset_num_proc
                )
                raw_testset = raw_testset.map(
                    lambda x: {"chosen": reddit_prompt_template(x, "chosen"),
                            "rejected": reddit_prompt_template(x, "rejected")},
                    num_proc=config.dataset_num_proc
                )
                for name, raw_validset in fine_grained_validset.items():
                    fine_grained_validset[name] = raw_validset.map(
                        lambda x: {"chosen": reddit_prompt_template(x, "chosen"),
                                "rejected": reddit_prompt_template(x, "rejected")},
                        num_proc=config.dataset_num_proc
                    )
                
                # Remove unnecessary columns (specific to the Reddit TL;DR dataset)
                raw_trainset = raw_trainset.remove_columns(["info", "summaries", "batch", "split", "extra"])
                raw_testset = raw_testset.remove_columns(["info", "summaries", "batch", "split", "extra"])
                for name, raw_validset in fine_grained_validset.items():
                    fine_grained_validset[name] = raw_validset.remove_columns(["info", "summaries", "batch", "split", "extra"])
        
        if not args.apply_chat_template:
            # Tokenize inputs
            raw_trainset = raw_trainset.map(
                preprocess_function,
                batched=True,
                num_proc=config.dataset_num_proc,
            )
            raw_testset = raw_testset.map(
                preprocess_function,
                batched=True,
                num_proc=config.dataset_num_proc,
            )
            for name, raw_validset in fine_grained_validset.items():
                fine_grained_validset[name] = raw_validset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=config.dataset_num_proc,
                )
            
            # Select the labeler
            if args.selected_labeler == "all" or args.selected_labeler == "personalized":
                pass
            else:
                print(f"Selecting labeler: {args.selected_labeler}")
                raw_trainset = raw_trainset.filter(lambda x: x["worker"] == int(args.selected_labeler))
                raw_testset = raw_testset.filter(lambda x: x["worker"] == int(args.selected_labeler))
            
            # TODO: Filter out examples that are too long
            # shuffle the dataset
            raw_trainset = raw_trainset.shuffle(seed=42)
            raw_testset = raw_testset.shuffle(seed=42)
            for name, raw_validset in fine_grained_validset.items():
                fine_grained_validset[name] = raw_validset.shuffle()

    if args.apply_chat_template:
        train_dataset = raw_datasets[args.dataset_train_split]
        eval_dataset = raw_datasets[args.dataset_test_split]
    else:
        train_dataset = raw_trainset
        if args.selected_labeler in ["all", "personalized"]:
            eval_dataset = fine_grained_validset
            # eval_dataset["all"] = raw_testset
            # eval_dataset = raw_testset
        else:
            eval_dataset = raw_testset
    print("Train Dataset: ", train_dataset)
    print("Eval Dataset: ", eval_dataset)

    return train_dataset, eval_dataset