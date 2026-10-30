from accelerate import Accelerator
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class EarlyStopBase(ABC):
    @abstractmethod
    def check_early_stopping(self, current_loss):
        pass

    @abstractmethod
    def is_best_overall(self):
        pass

class MultiMetricEarlyStopping:
    def __init__(self, metrics_config: Dict[str, List[Tuple[str, str, float]]], patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.metrics_config = {
            task: {name: {'mode': mode, 'weight': weight} for name, mode, weight in config}
            for task, config in metrics_config.items()
        }
        self.counter = {
            task: {name: 0 for name in task_config}
            for task, task_config in self.metrics_config.items()
        }
        self.best_score = {
            task: {name: float('inf') if config['mode'] == 'min' else float('-inf') 
                   for name, config in task_config.items()}
            for task, task_config in self.metrics_config.items()
        }
        self.total_weight = sum(sum(metric['weight'] for metric in task_config.values())
                                for task_config in self.metrics_config.values())
        self.latest_scores = {}

    def check_early_stopping(self, metrics: Dict[str, Dict[str, float]]) -> bool:
        missing_tasks = [task for task in metrics if task not in self.metrics_config]
        if missing_tasks:
            print(f"Warning EarlyStopping: The following tasks are not in the initialized metrics_config and will be ignored: {missing_tasks}")

        votes = 0
        self.latest_scores = {}
        for task, task_metrics in metrics.items():
            if task not in self.metrics_config:
                continue

            self.latest_scores[task] = {}
            for metric, value in task_metrics.items():
                if metric not in self.metrics_config[task]:
                    print(f"Warning EarlyStopping: Metric '{metric}' for task '{task}' is not in metrics_config and will be ignored.")
                    continue

                config = self.metrics_config[task][metric]
                mode = config['mode']
                weight = config['weight']
                self.latest_scores[task][metric] = value

                if mode == 'min':
                    if value < self.best_score[task][metric] - self.min_delta:
                        self.best_score[task][metric] = value
                        self.counter[task][metric] = 0
                    else:
                        self.counter[task][metric] += 1
                else:  # mode == 'max'
                    if value > self.best_score[task][metric] + self.min_delta:
                        self.best_score[task][metric] = value
                        self.counter[task][metric] = 0
                    else:
                        self.counter[task][metric] += 1

                if self.counter[task][metric] >= self.patience:
                    votes += weight

        return votes > self.total_weight / 2

    def is_best_overall(self) -> bool:
        if not self.latest_scores:
            return False

        best_count = 0
        total_count = 0

        for task, task_metrics in self.latest_scores.items():
            for metric, value in task_metrics.items():
                config = self.metrics_config[task][metric]
                mode = config['mode']
                weight = config['weight']

                if mode == 'min':
                    if value <= self.best_score[task][metric]:
                        best_count += weight
                else:  # mode == 'max'
                    if value >= self.best_score[task][metric]:
                        best_count += weight
                
                total_count += weight

        return best_count > total_count / 2

    def get_best_metrics(self) -> Dict[str, Dict[str, float]]:
        return {task: task_scores.copy() for task, task_scores in self.best_score.items()}

    def reset(self):
        self.counter = {
            task: {name: 0 for name in task_config}
            for task, task_config in self.metrics_config.items()
        }
        self.best_score = {
            task: {name: float('inf') if config['mode'] == 'min' else float('-inf') 
                   for name, config in task_config.items()}
            for task, task_config in self.metrics_config.items()
        }
        self.latest_scores = {}

    def save_state(self) -> Dict[str, Any]:
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'metrics_config': self.metrics_config,
            'counter': self.counter,
            'best_score': self.best_score,
            'total_weight': self.total_weight,
            'latest_scores': self.latest_scores
        }

    def load_state(self, state: Dict[str, Any]):
        self.patience = state['patience']
        self.min_delta = state['min_delta']
        self.metrics_config = state['metrics_config']
        self.counter = state['counter']
        self.best_score = state['best_score']
        self.total_weight = state['total_weight']
        self.latest_scores = state['latest_scores']