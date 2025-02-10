from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect 
import numpy as np

from quickstats import AbstractObject
from quickstats.core.hashing import hash_dict
from sklearn.metrics import accuracy_score, roc_curve, log_loss, auc

from .metrics import sic, max_sic, threshold_sic, nll

class ModelOutput(AbstractObject):
    """A class for computing and caching various model evaluation metrics.
    
    This class handles the computation of metrics for model outputs,
    including but not limited to classification metrics like ROC curves,
    regression metrics, and custom metrics like significance scores.
    Provides caching functionality to avoid redundant computations.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Model predictions or scores
    weight : Optional[np.ndarray], default=None
        Sample weights
    cache : bool, default=True
        Whether to cache computed metrics
    verbosity : str, default='INFO'
        Logging verbosity level
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: Optional[np.ndarray] = None,
        cache: bool = True,
        verbosity: Optional[str] = 'INFO'
    ) -> None:
        super().__init__(verbosity=verbosity)
        self._data = {}
        self._cached_metrics = {}
        self.cache = cache
        self.set_data(y_true, y_pred, weight=weight)

    @property
    def data(self) -> Dict[str, np.ndarray]:
        """Get the stored data dictionary.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing y_true, y_pred, and sample_weight arrays
        """
        return self._data

    @property
    def sample_size(self) -> int:
        if not self._data:
            return 0
        return self._data['y_pred'].shape[0]

    def set_data(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: Optional[np.ndarray] = None
    ) -> None:
        """Set or update the input data for metric calculations.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Model predictions or scores
        weight : Optional[np.ndarray], default=None
            Sample weights
        """
        data = {
            'y_true': y_true,
            'y_pred': y_pred,
            'sample_weight': weight
        }
        for key, array in data.items():
            if array is None:
                continue
            array = np.array(array)
            if (np.ndim(array) == 2) and (array.shape[-1] == 1):
                array = array.flatten()
            data[key] = array
        self._data = data

    def clear_cache(self) -> None:
        self._cached_metrics = {}

    def _evaluate(
        self,
        names: List[str],
        evaluator: Callable,
        kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Evaluate metrics using provided evaluator.

        Parameters
        ----------
        names : List[str]
            Names of metrics to compute
        evaluator : Callable
            Function to compute the metrics
        kwargs : Dict[str, Any]
            Arguments to pass to the evaluator

        Returns
        -------
        Tuple[Any, ...]
            Computed metric values

        Raises
        ------
        RuntimeError
            If number of results doesn't match number of metric names
        """
        results = evaluator(**kwargs)
        results = (results,) if len(names) == 1 else results
        
        if not isinstance(results, tuple) or len(names) != len(results):
            raise RuntimeError('Number of return arguments does not match number of requested metrics')
            
        return results

    def _store(
        self,
        names: List[str],
        kwargs_hash: int,
        results: Tuple[Any, ...]
    ) -> None:
        """Store results in cache.

        Parameters
        ----------
        names : List[str]
            Names of metrics to store
        kwargs_hash : int
            Hash value of the kwargs used
        results : Tuple[Any, ...]
            Results to store
        """
        for name, result in zip(names, results):
            self._cached_metrics[(name, kwargs_hash)] = result

    def _has_cache(self, names: List[str], kwargs_hash: int) -> bool:
        """Check if all requested metrics are in cache.

        Parameters
        ----------
        names : List[str]
            Names of metrics to check
        kwargs_hash : int
            Hash value of the kwargs used

        Returns
        -------
        bool
            True if all metrics are cached
        """
        return all((name, kwargs_hash) in self._cached_metrics for name in names)

    def _retrieve_cached(
        self,
        names: List[str],
        kwargs_hash: int
    ) -> Union[Any, Tuple[Any, ...]]:
        """Retrieve metrics from cache.

        Parameters
        ----------
        names : List[str]
            Names of metrics to retrieve
        kwargs_hash : int
            Hash value of the kwargs used

        Returns
        -------
        Union[Any, Tuple[Any, ...]]
            Cached metric value(s)
        """
        self.stdout.debug(f"Cached values for the metrics {', '.join(names)}")
        results = tuple(self._cached_metrics[(name, kwargs_hash)] for name in names)
        return results[0] if len(names) == 1 else results
        
    def _retrieve(
        self,
        names: Union[List[str], str],
        evaluator: Callable,
        major_keys: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Any:
        """Retrieve metrics, computing them if not cached.

        Parameters
        ----------
        names : Union[List[str], str]
            Name(s) of metrics to retrieve
        evaluator : Callable
            Function to compute metrics if not cached
        major_keys : Optional[List[str]], default=None
            Keys to exclude from cache key computation
        **kwargs : Any
            Additional arguments for metric computation

        Returns
        -------
        Union[Any, Tuple[Any, ...]]
            Requested metric value(s)
        """
        names = [names] if isinstance(names, str) else names
        major_keys = major_keys or []
        
        cache_kwargs = {k: v for k, v in kwargs.items() if k not in major_keys}
        kwargs_hash = hash_dict(cache_kwargs)
        
        if self.cache and self._has_cache(names, kwargs_hash):
            return self._retrieve_cached(names, kwargs_hash)
            
        results = self._evaluate(names, evaluator, kwargs)

        if self.cache:
            self._store(names, kwargs_hash, results)
            
        return results[0] if len(names) == 1 else results

    def _update_kwargs(self, names: List[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Update kwargs with stored data for specified names.

        Parameters
        ----------
        names : List[str]
            Names of data items to include
        kwargs : Dict[str, Any]
            Original kwargs dictionary

        Returns
        -------
        Dict[str, Any]
            Updated kwargs dictionary
        """
        kwargs_ = {name: self.data[name] for name in names}
        kwargs_.update(kwargs)
        return kwargs_

    def log_loss(self, **kwargs: Any) -> float:
        """Compute log loss (cross-entropy loss).

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to sklearn.metrics.log_loss

        Returns
        -------
        float
            Log loss value
        """
        kwargs = self._update_kwargs(['y_true', 'y_pred', 'sample_weight'], kwargs)
        if len(np.unique(kwargs['y_true'])) == 1:
            kwargs['labels'] = [0, 1]
        return self._retrieve(
            'log_loss',
            log_loss,
            ['y_true', 'y_pred', 'sample_weight'],
            **kwargs
        )
        
    def roc_curve(self, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Receiver Operating Characteristic (ROC) curve.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to sklearn.metrics.roc_curve

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (fpr, tpr, thresholds)
        """
        kwargs = self._update_kwargs(['y_true', 'y_pred', 'sample_weight'], kwargs)
        kwargs['y_score'] = kwargs.pop('y_pred')
        return self._retrieve(
            ['fpr', 'tpr', 'thresholds'],
            roc_curve,
            ['y_true', 'y_score', 'sample_weight'],
            **kwargs
        )

    def auc(self, **kwargs: Any) -> float:
        """Compute Area Under the Curve (AUC) score.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to sklearn.metrics.auc

        Returns
        -------
        float
            AUC score
        """
        fpr, tpr, thresholds = self.roc_curve(**kwargs)
        return self._retrieve(
            'auc',
            auc,
            ['x', 'y'],
            x=fpr,
            y=tpr,
            **kwargs
        )
        
    def sic(
        self,
        epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> float:
        """Compute SIC.

        Parameters
        ----------
        epsilon : Optional[float], default=None
            Epsilon parameter for SIC calculation
        **kwargs : Any
            Additional arguments for ROC curve computation

        Returns
        -------
        float
            SIC
        """
        fpr, tpr, thresholds = self.roc_curve(**kwargs)
        return self._retrieve(
            'sic',
            sic,
            ['tpr', 'fpr'],
            tpr=tpr,
            fpr=fpr,
            epsilon=epsilon
        )
        
    def max_sic(
        self,
        epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> float:
        """Compute maximum SIC.

        Parameters
        ----------
        epsilon : Optional[float], default=None
            Epsilon parameter for sic calculation
        **kwargs : Any
            Additional arguments for ROC curve computation

        Returns
        -------
        float
            Maximum SIC
        """
        fpr, tpr, thresholds = self.roc_curve(**kwargs)
        return self._retrieve(
            'max_sic',
            max_sic,
            ['fpr', 'tpr'],
            tpr=tpr,
            fpr=fpr,
            epsilon=epsilon
        )

    def threshold_sic(
        self,
        fpr_thres: float,
        **kwargs: Any
    ) -> float:
        """Compute threshold SIC.

        Parameters
        ----------
        fpr_thres : float
            False positive rate threshold
        **kwargs : Any
            Additional arguments for ROC curve computation

        Returns
        -------
        float
            Threshold SIC
        """
        fpr, tpr, thresholds = self.roc_curve(**kwargs)
        return self._retrieve(
            'threshold_sic',
            threshold_sic,
            ['fpr', 'tpr'],
            tpr=tpr,
            fpr=fpr,
            fpr_thres=fpr_thres
        )
    
    def nll(self, **kwargs: Any) -> float:
        """Compute negative log-likelihood.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for NLL computation

        Returns
        -------
        float
            Negative log-likelihood value
        """
        kwargs = self._update_kwargs(['y_true', 'y_pred', 'sample_weight'], kwargs)
        return self._retrieve(
            'nll',
            nll,
            ['y_true', 'y_pred', 'sample_weight'],
            **kwargs
        )