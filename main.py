import argparse
import json
import numpy as np
import random
import itertools

from lm_eval import models, tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--run_mc_validation', action="store_true")
    parser.add_argument('--limit', default=None)
    return parser.parse_args()

def main():
    lm = models.get_model(args.model).create_from_arg_string(args.model_args)
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)
    results = {}
    for task_name, task in task_dict.items():
        if not task.has_validation_docs():
            continue
        result = task.evaluate(
            docs=itertools.isslice(task.validation_docs(), 0, args.limit),
            lm=lm,
            provide_description=args.provide_description,
            num_fewshot=args.num_fewshot,
        )
        results[task_name] = result

    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        with open(args.output_path+task_names[0]+'.jsonl', "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_path = "./outputs/"
    main(args)
