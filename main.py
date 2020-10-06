import argparse
import json
import numpy as np
import random

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
    return parser.parse_args()

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def cross_validation(args, training_docs, task, shuffled_train_indices=None):
    lm = models.get_model(args.model).create_from_arg_string(args.model_args)
    results = []

    if shuffled_train_indices is None:
        shuffled_train_indices= [*range(len(training_docs))]
    random.shuffle(shuffled_train_indices)
    for i in range(0, len(shuffled_train_indices), args.num_fewshot):
        train_indices = shuffled_train_indices[i: i+args.num_fewshot]
        train_subset = []
        val_subset = []
        for i in range(len(shuffled_train_indices)):
            if i in train_indices:
                train_subset.append(training_docs[i])
            else:
                val_subset.append(training_docs[i])
        print(len(val_subset))

        accuracy = task.evaluate(val_subset, lm, args.provide_description, args.num_fewshot, train_doc=train_subset)

        result = {}
        result['indices'] = train_indices
        result['accuracy'] = accuracy
        results.append(result)
    return results

def mc_cross_validation(args, training_docs, task, shuffled_train_indices=None):
    num_cross_validation = 10
    all_cross_validation_results = np.zeros((len(training_docs), num_cross_validation))
    for i in range(num_cross_validation):
        results = cross_validation(args, training_docs, task, shuffled_train_indices)
        for tr_subset in results:
            all_cross_validation_results[tr_subset["indices"], i] = tr_subset['accuracy']['major']
    item_scores = all_cross_validation_results.mean(axis=1)
    return all_cross_validation_results, item_scores

def cross_validation_main(args):

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)
    task = task_dict['copa']
    training_docs = task.training_docs()

    all_cross_validation_results, item_scores = mc_cross_validation(args, training_docs, task)
    best_k = largest_indices(item_scores, 100)[0]
    print(best_k)
    print(item_scores)
    all_cross_validation_results, item_scores = mc_cross_validation(args, training_docs, task, best_k.tolist())
    best_k = largest_indices(item_scores, 32)[0]
    print(best_k)
    print(item_scores)

    train_subset = []
    for i in range(len(training_docs)):
        if i in best_k:
            train_subset.append(training_docs[i])
    lm = models.get_model(args.model).create_from_arg_string(args.model_args)
    accuracy = task.evaluate(task.validation_docs(), lm, args.provide_description, args.num_fewshot, train_doc=train_subset)
    print("final accuracy: ", accuracy)

def main(args):
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
            docs=task.validation_docs(),
            lm=lm,
            provide_description=args.provide_description,
            num_fewshot=args.num_fewshot,
        )
        results[task_name] = result

    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.run_mc_validation:
        print("running MCCV")
        cross_validation_main(args)
    else:
        print("running main")
        main(args)
