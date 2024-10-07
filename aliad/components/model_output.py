from typing import Optional, List
import inspect 

import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, log_loss, auc

from quickstats import AbstractObject
from aliad.components.metrics import significance, threshold_significance, negative_log_likelihood, nll, prior_ratio

class ModelOutput(AbstractObject):

    @property
    def data(self):
        return self._data

    @property
    def metrics(self):
        return self._metrics

    @property
    def cond_metrics(self):
        return self._cond_metrics    

    def __init__(self, y_true:np.ndarray, y_score:np.ndarray,
                 weight:Optional[np.ndarray]=None,
                 verbosity:Optional[str]='INFO'):
        super().__init__(verbosity=verbosity)
        self.set_data(y_true, y_score, weight=weight)

    def set_data(self, y_true:np.ndarray, y_score:np.ndarray,
                 weight:Optional[np.ndarray]=None):
        self.reset()
        data = {
            'y_true': y_true,
            'y_score': y_score,
            'sample_weight': weight
        }
        for key, array in data.items():
            if array is None:
                continue
            array = np.array(array)
            if (np.ndim(array) == 2) and (array.shape[-1] == 1):
                array = array.flatten()
            data[key] = array
        # TODO: assert binary label
        data['y_pred'] = np.round(data['y_score'])
        self._data = data
        
    def reset(self):
        self._data = {}
        self._metrics = {}
        self._cond_metrics = {}
        
    def _register_metrics(self, name:str, value):
        self._metrics[name] = value

    def _register_cond_metrics(self, name:str, value, cond):
        if (name not in self.cond_metrics):
            self._cond_metrics[name] = {}
        self._cond_metrics[name][cond] = value

    def _retrieve_metrics(self, name:str, evaluator, cache:bool=True, **kwargs):
        if cache and (name in self.metrics):
            self.stdout.debug(f"Cached value for the metric {name}")
            return self.metrics[name]
        value = evaluator(**kwargs)
        self._register_metrics(name, value=value)
        return value

    def _retrieve_cond_metrics(self, name:str, evaluator, cond,
                               cache:bool=True, **kwargs):
        if cache and ((name in self.cond_metrics) and (cond in self.cond_metrics[name])):
            self.stdout.debug(f"Cached value for the metric {name}")
            return self.cond_metrics[name][cond]
        value = evaluator(**kwargs)
        self._register_cond_metrics(name, value=value, cond=cond)
        return value

    def _retrieve_group_metrics(self, names:List[str], evaluator, cache:bool=True, **kwargs):
        if cache and all(name in self.metrics for name in names):
            self.stdout.debug(f"Cached values for the metrics {', '.join(names)}")
            return tuple(self.metrics[name] for name in names)
        values = evaluator(**kwargs)
        if len(names) != len(values):
            raise RuntimeError('number of return arguments does not match number of requested metrics')
        for name, value in zip(names, values):
            self._register_metrics(name, value=value)
        return values

    def _retrieve_group_cond_metrics(self, names:List[str], evaluator, cond,
                                     cache:bool=True, **kwargs):
        if (cache and 
            all(name in self.cond_metrics for name in names) and
            all(cond in self.cond_metrics[name] for name in names)):
            self.stdout.debug(f"Cached values for the metrics {', '.join(names)}")
            return tuple(self.cond_metrics[name][cond] for name in names)
        values = evaluator(**kwargs)
        if len(names) != len(values):
            raise RuntimeError('number of return arguments does not match number of requested metrics')
        for name, value in zip(names, values):
            self._register_cond_metrics(name, value=value, cond=cond)
        return values    

    def _update_kwargs(self, names:List[str], kwargs):
        kwargs_ = {name: self.data[name] for name in names}
        kwargs_.update(kwargs)
        return kwargs_

    def s_over_b(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true'], kwargs)
        def get_s_over_b(y_true):
            s = np.sum(y_true == 1)
            b = np.sum(y_true == 0)
            return s / b
        return self._retrieve_metrics('s_over_b', get_s_over_b, cache=cache,
                                      **kwargs)
        
    def s_over_sqrt_b(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true'], kwargs)
        def get_s_over_sqrt_b(y_true):
            s = np.sum(y_true == 1)
            b = np.sum(y_true == 0)
            return s / np.sqrt(b)
        return self._retrieve_metrics('s_over_sqrt_b', get_s_over_sqrt_b, cache=cache,
                                      **kwargs)

    def roc_curve(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true', 'y_score', 'sample_weight'], kwargs)
        return self._retrieve_group_metrics(['fpr', 'tpr', 'thresholds'],
                                            roc_curve, cache=cache,
                                            **kwargs)

    def negative_log_likelihood(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true', 'y_score', 'sample_weight'], kwargs)
        kwargs['y_pred'] = kwargs.pop('y_score')
        return self._retrieve_metrics('negative_log_likelihood', negative_log_likelihood,
                                      cache=cache, **kwargs)

    def nll(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true', 'y_score', 'sample_weight'], kwargs)
        kwargs['y_pred'] = kwargs.pop('y_score')
        return self._retrieve_metrics('nll', nll, cache=cache, **kwargs)
        
    def log_loss(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true', 'y_score', 'sample_weight'], kwargs)
        kwargs['y_pred'] = kwargs.pop('y_score')
        if len(np.unique(kwargs['y_true'])) == 1:
            kwargs['labels'] = [0, 1]
        return self._retrieve_metrics('log_loss', log_loss, cache=cache, **kwargs)

    def auc(self, cache:bool=True, **kwargs):
        fpr, tpr, thresholds = self.roc_curve(cache=cache, **kwargs)
        return self._retrieve_metrics('auc', auc, cache=cache,
                                      x=fpr, y=tpr)

    def get(self, metric:str, cache:bool=True, **kwargs):
        if not hasattr(self, metric):
            raise RuntimeError(f'metric "{metric}" is not supported')
        method = getattr(self, metric)
        return method(cache=cache, **kwargs)

    def significance(self, cache:bool=True, epsilon:Optional[float]=None, **kwargs):
        fpr, tpr, thresholds = self.roc_curve(cache=cache, **kwargs)
        return self._retrieve_cond_metrics('significance', significance, cache=cache,
                                           cond=epsilon, tpr=tpr, fpr=fpr, epsilon=epsilon)
        
    def max_significance(self, cache:bool=True, epsilon:Optional[float]=None, **kwargs):
        fpr, tpr, thresholds = self.roc_curve(cache=cache, **kwargs)
        return self._retrieve_cond_metrics('max_significance', max_significance, cache=cache,
                                           cond=epsilon, tpr=tpr, fpr=fpr, epsilon=epsilon)

    def threshold_significance(self, fpr_thres, cache:bool=True, **kwargs):
        fpr, tpr, thresholds = self.roc_curve(cache=cache, **kwargs)
        return self._retrieve_cond_metrics('threshold_significance', threshold_significance, cache=cache,
                                           cond=fpr_thres, tpr=tpr, fpr=fpr, fpr_thres=fpr_thres)

    def prior_ratio(self, cache:bool=True, **kwargs):
        kwargs = self._update_kwargs(['y_true', 'y_score', 'sample_weight'], kwargs)
        kwargs['y_pred'] = kwargs.pop('y_score')
        return self._retrieve_metrics('prior_ratio', prior_ratio, cache=cache, **kwargs)