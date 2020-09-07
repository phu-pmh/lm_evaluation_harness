import importlib
import os
from ..base import Registry

TASK_REGISTRY = Registry(registry_name="tasks")
# Load all modules in models directory to populate registry
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('lm_evaluation_harness.tasks.' + module_name)


def get_task(model_name):
    return TASK_REGISTRY.registry[model_name]