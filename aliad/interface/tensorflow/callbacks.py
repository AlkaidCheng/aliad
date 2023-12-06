import os
import json
from typing import Any, Optional, Union

import tensorflow as tf

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler for the Adam optimizer in TensorFlow.

    Parameters:
    initial_lr (float): Initial learning rate.
    lr_decay_factor (float): Decay factor applied to the learning rate.
    patience (int): Number of epochs with no improvement in validation loss before reducing the learning rate.
    min_lr (float): Minimum learning rate allowed.
    verbose (bool): If True, print updates about learning rate changes.
    """
    def __init__(self, initial_lr=0.001, lr_decay_factor=0.5, patience=10, min_lr=1e-7, verbose=False):
        super(LearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.lr_decay_factor = lr_decay_factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.initial_lr * self.lr_decay_factor, self.min_lr)
                if self.verbose:
                    print(f"\nEpoch {epoch + 1}: Reducing learning rate to {new_lr}")
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0
                self.best_loss = current_loss


class BatchMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BatchMetricsCallback, self).__init__()
        self.batch_train_metrics = []
        self.batch_val_metrics = []

    def on_train_batch_end(self, batch, logs=None):
        if logs:
            self.batch_train_metrics.append(logs.copy())

    def on_test_batch_end(self, batch, logs=None):
        if logs:
            self.batch_val_metrics.append(logs.copy())
            
class MetricsLogger(tf.keras.callbacks.Callback):

    """
    A TensorFlow Keras callback to log and save training and testing metrics.

    Provides detailed logs of metrics for each epoch and batch during training 
    and evaluation of a TensorFlow model.

    Parameters:
        filepath (str): Directory where metrics log files will be saved. Defaults to './logs'.
        save_freq (Union[str, int]): Determines the frequency of saving logged metrics. Defaults to -1.
            - If 'epoch', saves epoch-level metrics at the end of each epoch.
            - If 'batch', saves batch-level metrics after every training/testing batch.
            - If a positive integer, saves accumulated batch-level metrics at this interval.
            - If a negative integer, saves accumulated batch-level metrics over all batches at the end of each epoch.
    """
    
    TRAIN = 'train'
    TEST = 'test'

    def __init__(
        self,
        filepath: str = './logs',
        save_freq: Union[str, int] = -1,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        if save_freq == "batch":
            save_freq = 1
        if (save_freq != "epoch") and not isinstance(save_freq, int):
            raise ValueError('save_freq must be "epoch", "batch" or an integer')

        self.save_batch = isinstance(save_freq, int)
        self.save_freq = save_freq if self.save_batch else None
        self.filepath = filepath
        self._current_epoch = 0
        self.reset_batch_data()    

    def reset_batch_data(self):
        self._current_batch = {}
        self._batch_logs = {}
        for stage in ['train', 'test']:
            self._current_batch[stage] = 0
            self._batch_logs[stage] = []

    def reset_batch_data(self):
        """Resets the batch data storage for a new epoch."""
        self._current_batch = {self.TRAIN: 0, self.TEST: 0}
        self._batch_logs = {self.TRAIN: [], self.TEST: []}

    def get_epoch_metrics_savedir(self):
        """Returns the directory path for saving epoch metrics."""
        return os.path.join(self.filepath, "epoch_metrics")

    def get_batch_metrics_savedir(self):
        """Returns the directory path for saving batch metrics."""
        return os.path.join(self.filepath, "batch_metrics")

    def _log_epoch(self, epoch, logs):
        """Logs epoch-level metrics."""
        logs = dict() if logs is None else dict(logs)
        logs["epoch"] = epoch
        self._save_metrics(logs, stage='epoch')

    def _log_batch(self, batch, logs, stage: str):
        """Logs batch-level metrics."""
        if not self.save_batch:
            return
        logs = dict() if logs is None else dict(logs)
        logs["epoch"] = self._current_epoch
        logs["batch"] = batch
        
        self._batch_logs[stage].append(logs)

        if (self.save_freq > 0) and ((batch + 1) % self.save_freq == 0):
            self._save_metrics(self._batch_logs[stage], stage=stage)
            self._batch_logs[stage] = []
            
    def on_train_begin(self, logs=None):
        """Sets up directories and data at the start of training."""
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(self.get_epoch_metrics_savedir(), exist_ok=True)
        if self.save_batch:
            os.makedirs(self.get_batch_metrics_savedir(), exist_ok=True)
            self.reset_batch_data()

    def on_epoch_begin(self, epoch, logs=None):
        """Updates the current epoch index at the start of each epoch."""
        self._current_epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        """Updates the current batch index for training at the beginning of each batch."""
        self._current_batch[self.TRAIN] = batch

    def on_test_batch_begin(self, batch, logs=None):
        """Updates the current batch index for testing at the beginning of each batch."""
        self._current_batch[self.TEST] = batch

    def on_epoch_end(self, epoch, logs=None):
        """Logs and saves metrics at the end of each epoch."""
        self._log_epoch(epoch, logs)
        if self.save_batch:
            for stage, batch_logs in self._batch_logs.items():
                if batch_logs:
                    self._save_metrics(batch_logs, stage=stage)
            self.reset_batch_data()
        
    def on_train_batch_end(self, batch, logs=None):
        """Logs metrics at the end of each training batch."""
        self._log_batch(batch, logs, self.TRAIN)

    def on_test_batch_end(self, batch, logs=None):
        """Logs metrics at the end of each testing batch."""
        self._log_batch(batch, logs, self.TEST)

    def _save_metrics(self, logs, stage=None, indent: int = 2):
        """
        Saves the metrics to a file.

        Args:
            logs (dict or list): Metrics to be saved.
            stage (str, optional): The training stage ('train', 'test' or 'epoch').
            indent (int): Indentation level for pretty-printing the JSON file.
        """
        if not logs:
            return
        if isinstance(logs, list):  # Batch logs
            epoch = logs[0]['epoch']
            batch_start = logs[0]['batch']
            batch_end = logs[-1]['batch']
            if batch_start == batch_end:
                batch_range = f"{batch_start:04d}"
            else:
                batch_range = f"{batch_start:04d}_{batch_end:04d}"
            filename = os.path.join(self.get_batch_metrics_savedir(),
                                    f"{stage}_metrics_epoch_{epoch:04d}_batch_{batch_range}.json")
        else:  # Epoch logs
            epoch = logs['epoch']
            filename = os.path.join(self.get_epoch_metrics_savedir(), 
                                    f"{stage}_metrics_epoch_{epoch:04d}.json")

        with open(filename, 'w') as f:
            json.dump(logs, f, indent=indent)