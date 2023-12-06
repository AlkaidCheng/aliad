from typing import Any, Optional

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
    
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

    def __init__(
        self,
        filepath: str = './logs',
        save_freq: Union[str, int] = "epoch",
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        if save_freq == "batch":
            save_freq = -1
        if （save_freq != "epoch") and not isinstance(save_freq, int):
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

    def on_train_begin(self, logs=None):
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(self.get_epoch_metrics_savedir(), exist_ok=True)
        if self.log_batch:
            os.makedirs(self.get_batch_metrics_savedir(), exist_ok=True)
            self.reset_batch_data()

    def get_epoch_metrics_savedir(self):
        return os.path.join(self.filepath, "epoch_metrics")

    def get_batch_metrics_savedir(self):
        return os.path.join(self.filepath, "batch_metrics")

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._log_epoch(epoch, logs)
        if self.log_batch:
            for stage, batch_logs in self._batch_logs.items():
                if batch_logs:
                    self._save_metrics(batch_logs, stage=stage)
            self.reset_batch_data()

    def _log_epoch(self, epoch, logs):
        logs = dict() if logs is None else dict(logs)
        logs["epoch"] = epoch
        self._save_metrics(logs, stage='epoch')

    def _log_batch(self, batch, logs, stage: str):
        logs = dict() if logs is None else dict(logs)
        logs["epoch"] = self._current_epoch
        logs["batch"] = batch
        
        self._batch_logs[stage].append(logs)

        if self.log_batch and (batch > 0) and ((batch + 1) % self.save_freq == 0):
            self._save_metrics(self._batch_logs[stage], stage=stage)
            self._batch_logs[stage] = []
            
    def on_train_batch_begin(self, batch, logs=None):
        self._current_batch['train'] = batch
        
    def on_train_batch_end(self, batch, logs=None):
        self._log_batch(batch, logs, 'train')

    def on_test_batch_begin(self, batch, logs=None):
        self._current_batch['test'] = batch
    
    def on_test_batch_end(self, batch, logs=None):
        self._log_batch(batch, logs, 'test')

    def _save_metrics(self, logs, stage=None, indent: int = 2):
        if not logs:
            return
        if isinstance(logs, list):  # Batch logs
            epoch = logs[0]['epoch']
            batch_start = logs[0]['batch']
            batch_end = logs[-1]['batch']
            if batch_start == batch_end:
                batch_range = f"{batch_start:04d}_{batch_end:04d}"
            else:
                batch_range = f"{batch_start:04d}"
            filename = os.path.join(self.get_batch_metrics_savedir(),
                                    f"{stage}_metrics_epoch_{epoch:04d}_batch_{batch_range}.json")
        else:  # Epoch logs
            epoch = logs['epoch']
            filename = os.path.join(self.get_epoch_metrics_savedir(), 
                                    f"{stage}_metrics_epoch_{epoch:04d}.json")

        with open(filename, 'w') as f:
            json.dump(logs, f, indent=indent)