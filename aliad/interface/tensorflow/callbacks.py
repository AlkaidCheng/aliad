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