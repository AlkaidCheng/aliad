from typing import Dict, Any

from quickstats.core.modules import (
    module_exists,
    get_module_version,
    ModuleNotFoundError,
    ModuleVersionError
)

from quickstats.core.versions import Version

class BackendMixin:
    """
    A mixin to manage backend selection and validation.

    This class allows setting a backend dynamically while ensuring that
    all required modules exist and their versions meet specified constraints.

    Attributes
    ----------
    BACKENDS : set
        A set of supported backends.
    BACKEND_REQUIRES : dict
        A dictionary specifying required modules and version constraints
        for each backend.

    Examples
    --------
    Defining a backend with module dependencies:

    >>> class ExampleBackend(BackendMixin):
    ...     BACKENDS = {"tensorflow", "pytorch"}
    ...     BACKEND_REQUIRES = {
    ...         "tensorflow": {
    ...             "modules": ["tensorflow", "numpy"],  # Must be installed, any version
    ...             "versions": {  # Only `tensorflow` and `numpy` have version constraints
    ...                 "tensorflow": {"minimum": "2.0.0", "maximum": "2.12.0"},
    ...                 "numpy": {"minimum": "1.18.0"}
    ...             }
    ...         },
    ...         "pytorch": {
    ...             "modules": ["torch"],  # Must be installed, any version
    ...             "versions": {  # Only `torch` has a minimum version requirement
    ...                 "torch": {"minimum": "1.8.0"}
    ...             }
    ...         }
    ...     }

    Initializing and using a backend:

    >>> obj = ExampleBackend("tensorflow")  # Checks that required modules exist
    >>> obj.set_backend("pytorch")  # Switch to PyTorch after validation

    Handling invalid backends:

    >>> try:
    ...     obj.set_backend("nonexistent_backend")
    ... except ModuleNotFoundError as e:
    ...     print(e)  # Expected: Required module 'nonexistent_backend' is not installed.

    Handling outdated module versions:

    >>> try:
    ...     obj.set_backend("tensorflow")  # Suppose installed tensorflow version is 1.5.0
    ... except ModuleVersionError as e:
    ...     print(e)  # Expected: Module 'tensorflow' requires at least version 2.0.0, but found 1.5.0.
    """

    BACKENDS = {}
    BACKEND_REQUIRES = {}  # Defines required modules & version constraints per backend

    def __init__(self, backend: str):
        """
        Initialize a backend after validating dependencies.

        Parameters
        ----------
        backend : str
            The backend to use.

        Raises
        ------
        ModuleNotFoundError
            If a required module for the backend is missing.
        ModuleVersionError
            If a required module does not meet version constraints.
        """
        self.set_backend(backend)

    def set_backend(self, backend: str):
        """
        Set a new backend dynamically after validating its existence and version.

        Parameters
        ----------
        backend : str
            The backend to use.

        Raises
        ------
        ModuleNotFoundError
            If a required module for the backend is missing.
        ModuleVersionError
            If a required module does not meet version constraints.

        Examples
        --------
        Switching from one backend to another:

        >>> obj = ExampleBackend("tensorflow")
        >>> obj.set_backend("pytorch")  # Successfully switches if PyTorch is installed
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Allowed backends: {self.BACKENDS}")
        self._validate_backend_exists(backend)
        self._validate_backend_version(backend)
        self.backend = backend

    def _backend_dispatch(self, method_name: str, *args, **kwargs):
        """
        Dynamically dispatch a method to the corresponding backend-specific implementation.

        Parameters
        ----------
        method_name : str
            The name of the method to dispatch.
        *args : tuple
            Arguments passed to the backend-specific method.
        **kwargs : dict
            Keyword arguments passed to the backend-specific method.

        Returns
        -------
        Any
            The result of the backend-specific method.

        Raises
        ------
        NotImplementedError
            If the requested method is not implemented for the current backend.
        """
        backend_method = f"_{method_name}_{self.backend}"
        if self._has_backend_method(backend_method):
            return getattr(self, backend_method)(*args, **kwargs)
        raise NotImplementedError(f"Method '{method_name}' is not implemented for backend '{self.backend}'.")

    def _has_backend_method(self, method_name: str) -> bool:
        """
        Check if a backend-specific method is implemented.

        Parameters
        ----------
        method_name : str
            The name of the method to check.

        Returns
        -------
        bool
            True if the method is implemented, False otherwise.
        """
        return hasattr(self, method_name)

    def _validate_backend_exists(self, backend: str) -> bool:
        """
        Check if all required modules for the backend exist.

        Parameters
        ----------
        backend : str
            The backend to validate.

        Returns
        -------
        bool
            True if all required modules exist.

        Raises
        ------
        ModuleNotFoundError
            If a required module is missing.

        Examples
        --------
        >>> obj = ExampleBackend("tensorflow")  # Ensures TensorFlow and NumPy exist
        """
        if backend not in self.BACKEND_REQUIRES:
            return True  # No specific module dependencies, assume valid

        backend_data = self.BACKEND_REQUIRES[backend]

        # Check modules that must exist
        for module in backend_data.get("modules", []):
            if not module_exists(module):
                raise ModuleNotFoundError(f"Required module '{module}' for backend '{backend}' is not installed.")
        return True

    def _validate_backend_version(self, backend: str) -> bool:
        """
        Check if all required modules for the backend meet version constraints.

        Parameters
        ----------
        backend : str
            The backend to validate.

        Returns
        -------
        bool
            True if all required modules meet version constraints.

        Raises
        ------
        ModuleVersionError
            If a module does not meet the required version constraints.

        Examples
        --------
        >>> obj = ExampleBackend("tensorflow")  # Ensures TensorFlow is at least v2.0.0
        """
        if backend not in self.BACKEND_REQUIRES:
            return True  # No specific version constraints

        backend_versions = self.BACKEND_REQUIRES[backend].get("versions", {})

        for module, constraints in backend_versions.items():
            installed_version = get_module_version(module)

            min_version = constraints.get("minimum")
            max_version = constraints.get("maximum")

            if min_version and installed_version < Version(min_version):
                raise ModuleVersionError(
                    f"Module '{module}' for backend '{backend}' requires at least version {min_version}, "
                    f"but found {installed_version}."
                )

            if max_version and installed_version > Version(max_version):
                raise ModuleVersionError(
                    f"Module '{module}' for backend '{backend}' requires at most version {max_version}, "
                    f"but found {installed_version}."
                )
        return True

class ConfigMixin:

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)