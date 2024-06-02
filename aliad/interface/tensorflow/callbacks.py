import os
import re
import sys
import glob
import json
from operator import itemgetter
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from quickstats.utils.common_utils import NpEncoder

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

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)

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
                current_lr = self.model.optimizer.lr
                new_lr = max(current_lr * self.lr_decay_factor, self.min_lr)
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

class WeightsLogger(tf.keras.callbacks.Callback):

    BATCH = 'batch'
    EPOCH = 'epoch'

    SUBDIRS = {
        BATCH : 'batch_weights',
        EPOCH : 'epoch_weights'
    }   
    
    def __init__(
        self,
        filepath: str = './logs',
        save_freq: Union[str, int] = -1,
        display_weight:bool=False,
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
        self.display_weight = display_weight
        self.reset()

    def reset(self):
        self._current_epoch = 0
        self._logs = {
            self.EPOCH: []
        }
        if self.save_batch:
            self._logs[self.BATCH] = []
            self.reset_batch_data()

    def reset_batch_data(self):
        """Resets the batch data storage for a new epoch."""
        self._current_batch = 0
        self._current_epoch_batch_logs = []

    def get_weights_savedir(self, stage:str):
        """Returns the directory path for saving epoch metrics."""
        return os.path.join(self.filepath, self.SUBDIRS[stage])

    def _log_epoch(self, epoch, weights):
        """Logs epoch-level metrics."""
        logs = {
            "weights": np.array(weights).flatten(),
            "epoch"  : epoch
        }
        if self.display_weight:
            print(f"\n[WeightLogger] Epoch {epoch}, Trainable Weights = {logs['weights']}")
        self._current_epoch_logs = logs
        self._save_logs(logs, stage=self.EPOCH)

    def _log_batch(self, batch, weights):
        """Logs batch-level metrics."""
        if not self.save_batch:
            return
        logs = {
            "weights": np.array(weights).flatten(),
            "epoch"  : self._current_epoch,
            "batch"  : batch
        }

        self._current_epoch_batch_logs.append(logs)

        if (self.save_freq > 0) and ((batch + 1) % self.save_freq == 0):
            self._save_logs(self._current_epoch_batch_logs, stage=self.BATCH)
            self._logs[self.BATCH].extend(self._current_epoch_batch_logs)
            self._current_epoch_batch_logs= []

    def _update_logs(self):
        self._logs[self.EPOCH].append(self._current_epoch_logs)
        if self.save_batch:
            logs = self._current_epoch_batch_logs
            if logs:
                self._logs[self.BATCH].extend(logs)

    def on_train_begin(self, logs=None):
        """Sets up directories and data at the start of training."""
        try:
            trainable_weights = np.array(self.model.trainable_weights)
        except:
            raise RuntimeError("can not convert trainable weights into numpy arrays")
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(self.get_weights_savedir(self.EPOCH), exist_ok=True)
        if self.save_batch:
            os.makedirs(self.get_weights_savedir(self.BATCH), exist_ok=True)
        self.reset()

    def on_epoch_begin(self, epoch, logs=None):
        """Updates the current epoch index at the start of each epoch."""
        self._current_epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        """Updates the current batch index for training at the beginning of each batch."""
        self._current_batch = batch

    def on_epoch_end(self, epoch, logs=None):
        """Logs and saves weights at the end of each epoch."""
        trainable_weights = np.array(self.model.trainable_weights)
        self._log_epoch(epoch, trainable_weights)
        self._update_logs()
        if self.save_batch:
            if self._current_epoch_batch_logs:
                self._save_logs(self._current_epoch_batch_logs, stage=self.BATCH)
            self.reset_batch_data()
        
    def on_train_batch_end(self, batch, logs=None):
        """Logs weights at the end of each training batch."""
        trainable_weights = np.array(self.model.trainable_weights)
        self._log_batch(batch, trainable_weights)
    
    def _save_logs(self, logs, stage=None, indent: int = 2):
        """
        Saves the weight logs to a file.

        Args:
            logs (dict or list): Logs to be saved.
            stage (str, optional): The training stage ('batch' or 'epoch').
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
            filename = os.path.join(self.get_weights_savedir(stage),
                                    f"metrics_epoch_{epoch:04d}_batch_{batch_range}.json")
        else:  # Epoch logs
            epoch = logs['epoch']
            filename = os.path.join(self.get_weights_savedir(stage),
                                    f"metrics_epoch_{epoch:04d}.json")

        with open(filename, 'w') as f:
            json.dump(logs, f, indent=indent, cls=NpEncoder)

    def _get_logs_from_path(self, path:str):
        logs = []
        log_filenames = glob.glob(os.path.join(path, '*.json'))
        for filename in log_filenames:
            data = json.load(open(filename))
            if isinstance(data, list):
                logs.extend(data)
            else:
                logs.append(data)
        if logs:
            if 'batch' in logs[0]:
                logs = sorted(logs, key=itemgetter('epoch', 'batch'))
            else:
                logs = sorted(logs, key=itemgetter('epoch'))
        return logs

    def restore(self):
        self.reset()
        if self.save_batch:
            stages = [self.EPOCH, self.BATCH]
        else:
            stages = [self.EPOCH]
        for stage in stages:
            epoch_logs_path = self.get_metrics_savedir(stage)
            self._logs[stage] = self._get_logs_from_path(epoch_logs_path)
            
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
    EPOCH = 'epoch'

    SUBDIRS = {
        TRAIN : 'batch_train_metrics',
        TEST  : 'batch_test_metrics',
        EPOCH : 'epoch_metrics'
    }    

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
        self.reset()

    def reset(self):
        self._current_epoch = 0
        self._logs = {
            self.EPOCH: []
        }
        if self.save_batch:
            self._logs[self.TRAIN] = []
            self._logs[self.TEST] = []
            self.reset_batch_data()

    def reset_batch_data(self):
        """Resets the batch data storage for a new epoch."""
        self._current_batch = {self.TRAIN: 0, self.TEST: 0}
        self._current_epoch_batch_logs = {self.TRAIN: [], self.TEST: []}

    def get_metrics_savedir(self, stage:str):
        """Returns the directory path for saving epoch metrics."""
        return os.path.join(self.filepath, self.SUBDIRS[stage])

    def _log_epoch(self, epoch, logs):
        """Logs epoch-level metrics."""
        logs = dict() if logs is None else dict(logs)
        logs["epoch"] = epoch
        self._current_epoch_logs = logs
        self._save_metrics(logs, stage=self.EPOCH)

    def _log_batch(self, batch, logs, stage: str):
        """Logs batch-level metrics."""
        if not self.save_batch:
            return
        logs = dict() if logs is None else dict(logs)
        logs["epoch"] = self._current_epoch
        logs["batch"] = batch
        
        self._current_epoch_batch_logs[stage].append(logs)

        if (self.save_freq > 0) and ((batch + 1) % self.save_freq == 0):
            self._save_metrics(self._current_epoch_batch_logs[stage], stage=stage)
            self._logs[stage].extend(self._current_epoch_batch_logs[stage])
            self._current_epoch_batch_logs[stage] = []

    def _update_logs(self):
        self._logs[self.EPOCH].append(self._current_epoch_logs)
        if self.save_batch:
            for stage in [self.TRAIN, self.TEST]:
                logs = self._current_epoch_batch_logs[stage]
                if logs:
                    self._logs[stage].extend(logs)
            
    def on_train_begin(self, logs=None):
        """Sets up directories and data at the start of training."""
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(self.get_metrics_savedir(self.EPOCH), exist_ok=True)
        if self.save_batch:
            os.makedirs(self.get_metrics_savedir(self.TRAIN), exist_ok=True)
            os.makedirs(self.get_metrics_savedir(self.TEST), exist_ok=True)
        self.reset()

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
        self._update_logs()
        if self.save_batch:
            for stage, batch_logs in self._current_epoch_batch_logs.items():
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
            filename = os.path.join(self.get_metrics_savedir(stage),
                                    f"metrics_epoch_{epoch:04d}_batch_{batch_range}.json")
        else:  # Epoch logs
            epoch = logs['epoch']
            filename = os.path.join(self.get_metrics_savedir(stage),
                                    f"metrics_epoch_{epoch:04d}.json")

        with open(filename, 'w') as f:
            json.dump(logs, f, indent=indent)

    def _get_logs_from_path(self, path:str):
        logs = []
        log_filenames = glob.glob(os.path.join(path, '*.json'))
        for filename in log_filenames:
            data = json.load(open(filename))
            if isinstance(data, list):
                logs.extend(data)
            else:
                logs.append(data)
        if logs:
            if 'batch' in logs[0]:
                logs = sorted(logs, key=itemgetter('epoch', 'batch'))
            else:
                logs = sorted(logs, key=itemgetter('epoch'))
        return logs

    def restore(self):
        self.reset()
        if self.save_batch:
            stages = [self.EPOCH, self.TRAIN, self.TEST]
        else:
            stages = [self.EPOCH]
        for stage in stages:
            epoch_logs_path = self.get_metrics_savedir(stage)
            self._logs[stage] = self._get_logs_from_path(epoch_logs_path)

    def get_dataframe(self, stage:str, drop_first_batch:bool=True):
        if stage not in self._logs:
            return None
        import pandas as pd
        df = pd.DataFrame(self._logs[stage])
        if 'batch' in df.columns:
            df['epoch_cont'] = df['epoch'] + df['batch'] / (df['batch'].max() + 1)
            if drop_first_batch:
                df = df.loc[1:]
        return df
    
class EarlyStopping(tf.keras.callbacks.EarlyStopping):

    def __init__(
        self,
        *args,
        interrupt_freq:Optional[int]=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if (interrupt_freq) and (interrupt_freq <= 0):
            raise ValueError('interrupt_freq can not be negative')
        self.restore_config = {}
        self.interrupt_freq = interrupt_freq
        self.interrupted = False
        self.initial_epoch = 0
        self.final_epoch = 0
        
    def restore(self,
                model,
                metrics_ckpt_filepath:str,
                model_ckpt_filepath:str):
        epochs, metrics = self._get_metrics_ckpt_data(metrics_ckpt_filepath)
        if len(epochs) == 0:
            return None
        best_op = np.argmin if self.monitor_op == np.less else np.argmax
        best_idx = best_op(metrics)
        last_epoch = np.max(epochs)
        best_epoch = epochs[best_idx]
        best_metric = metrics[best_idx]
        model_filepath = self._get_model_filepath(model_ckpt_filepath,
                                                  epoch=best_epoch)
        model.load_weights(model_filepath)
        best_weights = model.get_weights()
        sys.stdout.write(f"[INFO] Found best metric value of {best_metric} from epoch {best_epoch}.\n")
        if best_epoch != last_epoch:
            model_filepath = self._get_model_filepath(model_ckpt_filepath,
                                                      epoch=last_epoch)
            model.load_weights(model_filepath)
        sys.stdout.write(f"[INFO] Restored model weights at epoch {last_epoch}"
                         f"{model_filepath}.\n")
            
        self.restore_config = {
            'wait': last_epoch - best_epoch,
            'best': best_metric,
            'best_weights': best_weights,
            'best_epoch': best_epoch,
            'stopped_epoch': 0
        }
        self.initial_epoch = last_epoch + 1
        
    def _get_metrics_ckpt_data(self, metrics_ckpt_filepath:str):
        path_wildcard = re.sub(r"{.*}", r"*", metrics_ckpt_filepath)
        ckpt_paths = glob.glob(path_wildcard)
        basename = os.path.basename(metrics_ckpt_filepath)
        basename_regex = re.compile("^" + re.sub(r"{.*}", r".*", basename) + "$")
        ckpt_paths = [path for path in ckpt_paths if basename_regex.match(os.path.basename(path))]
        epochs = []
        metrics = []                          
        for ckpt_path in ckpt_paths:
            with open(ckpt_path, "r") as ckpt_file:
                data = json.load(ckpt_file)
            epochs.append(data['epoch'])
            metrics.append(data[self.monitor])
        epochs = np.array(epochs)
        metrics = np.array(metrics)
        return epochs, metrics
        
    def _get_model_filepath(self, model_ckpt_filepath:str, epoch:int):
        filepath = model_ckpt_filepath.format(epoch=epoch+1)
        return filepath

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.restore_config:
            self.__dict__.update(self.restore_config)
            
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        self.final_epoch = epoch
        if (self.interrupt_freq and 
            ((epoch + 1 - self.initial_epoch) % self.interrupt_freq == 0)):
            self.model.stop_training = True
            self.interrupted = True

    def reset(self):
        if hasattr(self, 'model'):
            self.model.stop_training = False
        self.interrupted = False
        self.wait = 0