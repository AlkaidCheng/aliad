import os
import re
import sys
import glob
import json
from operator import itemgetter
from typing import Any, Optional, Union, Dict, Tuple
from collections import defaultdict

import numpy as np
import tensorflow as tf

from quickstats.utils.common_utils import (
    NpEncoder,
    list_of_dict_to_dict_of_list
)
from quickstats import DescriptiveEnum

from aliad.components.callbacks import LoggerSaveMode

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
        self.current_lr = None
        self.lr_decay_factor = lr_decay_factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best_loss = float('inf')
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def reset(self) -> None:
        self.current_lr = None
        self.wait = 0
        self.enabled = True

    def on_train_begin(self, logs=None):
        lr = self.current_lr or self.initial_lr
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return
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
                self.current_lr = new_lr
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


class BaseLogger(tf.keras.callbacks.Callback):

    BATCH_FILENAME = "Epoch_{epoch:04d}_Batch_{batch_start:05d}_{batch_end:05d}.json"
    EPOCH_FILENAME = "Epoch_{epoch_start:04d}_{epoch_end:04d}.json"

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_batch(self) -> int:
        return self._current_batch

    def __init__(
        self,
        dirname: str = "./logs",
        filename: Optional[str] = None,
        save_freq: Union[str, int, LoggerSaveMode] = "epoch",
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        self.set_save_freq(save_freq)
        self.set_paths(dirname, filename)
        self.reset_data()

    def set_save_freq(
        self,
        value: Union[str, int, LoggerSaveMode] = "epoch"
    ):
        if isinstance(value, int) and (value >= 0):
            save_mode = LoggerSaveMode.BATCH
            save_freq = value
        else:
            try:
                save_mode = LoggerSaveMode.parse(value)
                save_freq = -1
            except Exception as e:
                raise ValueError('Save frequency must be "train", "epoch", "batch" or non-negative integer')
        self.save_mode = save_mode
        self.save_freq = save_freq

    def set_paths(
        self,
        dirname: str = './logs',
        filename: Optional[str] = None
    ): 
        if filename is None:
            if self.save_mode == LoggerSaveMode.BATCH:
                filename = self.BATCH_FILENAME
            elif self.save_mode in [LoggerSaveMode.EPOCH, LoggerSaveMode.TRAIN]:
                filename = self.EPOCH_FILENAME
            else:
                raise ValueError(
                    f'Invalid save mode: {self.save_mode}'
                )
        self.dirname = dirname
        self.filename = filename

    def get_filepath(
        self,
        epoch_start: int,
        epoch_end: Optional[int] = None,
        batch_start: Optional[int] = None,
        batch_end: Optional[int] = None
    ) -> str:
        kwargs = {}
        if epoch_end is None:
            epoch_end = epoch_start
        kwargs['epoch_start'] = epoch_start
        kwargs['epoch_end'] = epoch_end
        kwargs['epoch'] = epoch_start
        if (batch_start is not None) and (batch_end is not None):
            kwargs['batch_start'] = batch_start
            kwargs['batch_end'] = batch_end
        basename = self.filename.format(**kwargs)
        return os.path.join(self.dirname, basename)

    def reset_data(self):     
        self.reset_epoch_data()
        self.reset_batch_data()

    def reset_epoch_data(self):
        self._current_epoch = 0
        self.reset_current_epoch_data()
        self._full_epoch_logs = defaultdict(list)

    def reset_current_epoch_data(self):
        self._current_epoch_logs = defaultdict(list)
        
    def reset_batch_data(self):
        self._current_batch = 0
        self.reset_current_batch_data()
        self._full_batch_logs = defaultdict(list)

    def reset_current_batch_data(self):
        self._current_batch_logs = defaultdict(list)

    def _update_logs(
        self,
        data: Dict[str, Any],
        epoch: int,
        batch: Optional[int] = None
    ) -> None:
        logs = self._current_epoch_logs if batch is None else self._current_batch_logs
        for key, value in data.items():
            logs[key].append(value)
        logs['epoch'].append(epoch)
        if batch is not None:
            logs['batch'].append(batch)

    def _extend_epoch_logs(self):
        for key, value in self._current_epoch_logs.items():
            self._full_epoch_logs[key].extend(value)
        self.reset_current_epoch_data()

    def _save_logs(
        self,
        mode: LoggerSaveMode,
        indent: int = 2,
    ) -> None:
        batch_start = None
        batch_end = None
        if mode == LoggerSaveMode.BATCH:
            logs = self._current_batch_logs
            batch_start = np.min(logs['batch'])
            batch_end = np.max(logs['batch'])
        elif mode == LoggerSaveMode.EPOCH:
            logs = self._current_epoch_logs
        elif mode == LoggerSaveMode.TRAIN:
            logs = self._full_epoch_logs
        else:
            raise ValueError(
                f'Invalid save mode: {mode}'
            )
        if not logs:
            return
        epoch_start = np.min(logs['epoch'])
        epoch_end = np.max(logs['epoch'])
        filepath = self.get_filepath(
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            batch_start=batch_start,
            batch_end=batch_end
        )
        with open(filepath, 'w') as file:
            json.dump(logs, file, indent=indent, cls=NpEncoder)

    def get_log_data(
        self,
        logs=None
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def on_train_begin(self, logs=None):
        """Sets up directories and data at the start of training."""
        os.makedirs(self.dirname, exist_ok=True)
        self.reset_data()

    def on_epoch_begin(self, epoch, logs=None):
        """Updates the current epoch index at the start of each epoch."""
        self._current_epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        """Updates the current batch index for training at the beginning of each batch."""
        self._current_batch = batch

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Logs and saves data at the end of each epoch."""
        data = self.get_log_data(logs)
        self._update_logs(data, epoch=epoch)

        if self.save_mode == LoggerSaveMode.EPOCH:
            self._save_logs(LoggerSaveMode.EPOCH)
            self._extend_epoch_logs()

        self.reset_batch_data()
        
    def on_train_batch_end(self, batch, logs=None) -> None:
        """Logs weights at the end of each training batch."""
        data = self.get_log_data(logs)
        self._update_logs(data, epoch=self.current_epoch, batch=batch)

        if self.save_mode == LoggerSaveMode.BATCH:
            if (self.save_freq > 0) and ((batch + 1) % self.save_freq == 0):
                self._save_logs(LoggerSaveMode.BATCH)
                self.reset_current_batch_data()

    def on_train_end(self, logs=None) -> None:
        if self.save_mode == LoggerSaveMode.TRAIN:
            self._extend_epoch_logs()
            self._save_logs(LoggerSaveMode.TRAIN)
        self.reset_current_epoch_data()
        self.reset_current_batch_data()

    def _get_logs_from_path(
        self,
        path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_logs = defaultdict(list)
        epoch_logs = defaultdict(list)
        filenames = glob.glob(os.path.join(path, '*.json'))
        for filename in filenames:
            with open(filename, 'r') as file:
                data = json.load(file)
            logs = batch_logs if 'batch' in data else epoch_logs
            for key, value in data.items():
                logs[key].extend(value)
        return batch_logs, epoch_logs

    def restore(self):
        self.reset_data()
        batch_logs = defaultdict(list)
        epoch_logs = defaultdict(list)
        filenames = glob.glob(os.path.join(self.dirname, '*.json'))
        for filename in filenames:
            with open(filename, 'r') as file:
                data = json.load(file)
            logs = batch_logs if 'batch' in data else epoch_logs
            for key, value in data.items():
                logs[key].extend(value)
        self._full_batch_logs = batch_logs
        self._full_epoch_logs = epoch_logs
        
class WeightsLogger(BaseLogger):
    
    def __init__(
        self,
        dirname: str = './weight_logs',
        filename: Optional[str] = None,
        save_freq: Union[str, int, LoggerSaveMode] = "epoch",
        display_weight: bool = False,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            dirname=dirname,
            filename=filename,
            save_freq=save_freq,
            *args, **kwargs
        )
        self.display_weight = display_weight

    def get_log_data(
        self,
        logs=None
    ) -> Dict[str, Any]:
        data = {
            'weights': np.array(self.model.trainable_weights)
        }
        return data

    def _update_logs(
        self,
        data: Dict[str, Any],
        epoch: int,
        batch: Optional[int] = None
    ) -> None:
        if (batch is None) and (self.display_weight):
            print(f"\n[WeightLogger] Epoch {epoch}, Trainable Weights = {data['weights']}")
        super()._update_logs(data=data, epoch=epoch, batch=batch)

    def on_train_begin(self, logs=None):
        """Sets up directories and data at the start of training."""
        try:
            trainable_weights = np.array(self.model.trainable_weights)
        except:
            raise RuntimeError("can not convert trainable weights into numpy arrays")
        super().on_train_begin(logs=logs)
            
class MetricsLogger(BaseLogger):

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
    """

    def __init__(
        self,
        dirname: str = './metric_logs',
        filename: Optional[str] = None,
        save_freq: Union[str, int, LoggerSaveMode] = "epoch",
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            dirname=dirname,
            filename=filename,
            save_freq=save_freq,
            *args, **kwargs
        )

    def get_log_data(
        self,
        logs=None
    ) -> Dict[str, Any]:
        data = dict() if logs is None else dict(logs)
        return data
    
class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(
        self,
        *args,
        interrupt_freq: Optional[int] = None,
        always_restore_best_weights: bool = False, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if (interrupt_freq) and (interrupt_freq <= 0):
            raise ValueError('interrupt_freq cannot be negative')
        self.restore_config = {} 
        self.interrupt_freq = interrupt_freq
        self.interrupted = False
        self.resumed = False
        self.initial_epoch = 0
        self.final_epoch = 0
        self.always_restore_best_weights = always_restore_best_weights  # New distinct flag

    def resume(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.resumed = True

    def restore(self, model, metrics_ckpt_filepath: str, model_ckpt_filepath: str):
        """
        Restore model weights from a checkpoint file using external metric tracking.
        """
        epochs, metrics = self._get_metrics_ckpt_data(metrics_ckpt_filepath)
        if len(epochs) == 0:
            return None

        best_op = np.argmin if self.monitor_op == np.less else np.argmax
        best_idx = best_op(metrics)
        last_epoch = np.max(epochs)
        best_epoch = epochs[best_idx]
        best_metric = metrics[best_idx]

        # Load best weights from external checkpoint
        model_filepath = self._get_model_filepath(model_ckpt_filepath, epoch=best_epoch)
        model.load_weights(model_filepath)
        best_weights = model.get_weights()
        sys.stdout.write(f"[INFO] Restored model weights at epoch {best_epoch} {model_filepath}.\n")
        #sys.stdout.write(f"[INFO] Found best metric value of {best_metric} from epoch {best_epoch}.\n")

        # Load final epoch weights for comparison
        #if best_epoch != last_epoch:
        #    model_filepath = self._get_model_filepath(model_ckpt_filepath, epoch=last_epoch)
        #    model.load_weights(model_filepath)
        #sys.stdout.write(f"[INFO] Restored model weights at epoch {last_epoch} {model_filepath}.\n")

        self.restore_config = {
            'wait': last_epoch - best_epoch,
            'best': best_metric,
            'best_weights': best_weights,
            'best_epoch': best_epoch,
            'stopped_epoch': 0
        }
        self.initial_epoch = last_epoch + 1

    def _get_metrics_ckpt_data(self, metrics_ckpt_filepath: str):
        """
        Load checkpointed metric values to determine the best epoch.
        """
        path_wildcard = re.sub(r"{.*}", r"*", metrics_ckpt_filepath)
        ckpt_paths = sorted(glob.glob(path_wildcard))
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

        epochs = np.array(epochs) + 1
        metrics = np.array(metrics)
        return epochs, metrics

    def _get_model_filepath(self, model_ckpt_filepath: str, epoch: int):
        """
        Generate the correct filepath for a given epoch checkpoint.
        """
        filepath = model_ckpt_filepath.format(epoch=epoch)
        return filepath

    def on_train_begin(self, logs=None):
        """
        Reset state and restore any previously saved training configuration.
        """
        if not self.resumed:
            super().on_train_begin(logs)
        if self.restore_config:
            self.__dict__.update(self.restore_config) 

    def on_epoch_end(self, epoch, logs=None):
        """
        Check if early stopping should be triggered or if training should be interrupted.
        """
        super().on_epoch_end(epoch, logs=logs)
        self.final_epoch = epoch
        if self.interrupt_freq and ((epoch + 1 - self.initial_epoch) % self.interrupt_freq == 0):
            self.model.stop_training = True
            self.interrupted = True

    def on_train_end(self, logs=None):
        """
        Ensures best weights are restored at the end of training if `always_restore_best_weights` is enabled,
        even if early stopping did not trigger.
        """
        super().on_train_end(logs)
        if self.always_restore_best_weights and self.best_weights is not None:
            if self.model.stop_training:
                text = "with"
            else:
                text = "without"
            sys.stdout.write(
                f"[INFO] Training completed {text} early stopping. Restoring best weights from epoch {self.best_epoch + 1}.\n"
            )
            self.model.set_weights(self.best_weights)

    def reset(self):
        """
        Reset early stopping tracking state.
        """
        if hasattr(self, 'model'):
            self.model.stop_training = False
        self.interrupted = False
        self.wait = 0
        self.resumed = False