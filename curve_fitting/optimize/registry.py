from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Type

if TYPE_CHECKING:
    from .optimizer import OptimizerBase


class Registry(Dict[str, Type]):
    """Registry of all available models"""

    def getName(self, model) -> str:
        """Get the name of the ssi object

        Parameters
        ----------
        model : MicroMechanicalModelBase
            The model object

        Returns
        -------
        str
            Name of the ssi object
        """
        for name, cls in self.items():
            if isinstance(model, cls):
                return name
        raise ValueError(f"Model {model} is not registered")


registry: Registry[str, Type["OptimizerBase"]] = Registry()  # noqa


def register(cls_or_name: Type | str = None, name: str = None, *, saveto: Registry = None) -> Type | Callable:
    """Register a new model

    Examples
    --------
    The following calls are equivalent:

    1) ``cls_or_name = "Foo"``, ``name = None``

       >>> @register("Foo")
       ... class Foo:
       ...     ...

    2) ``cls_or_name = None``, ``name = "Foo"``

       >>> @register(name="Foo")
       ... class Foo:
       ...     ...

    3) ``cls_or_name = Foo``, ``name = "Foo"``

       >>> class Foo:
       ...     ...
       >>> register(Foo, "Foo")
       <class 'micromechanical.registry.Foo'>

    Parameters
    ----------
    cls_or_name : Type | str, optional
        The class to register or the name of the class, by default None.
    name : str, optional
        The name of the class, by default None
    saveto : Registry, optional
        The registry to save the class, by default None which will use the default registry

    Returns
    -------
    Type | Callable
        The class that was registered or a callable that registers the class
    """
    saveto_registry = saveto if saveto is not None else registry
    cls = cls_or_name
    if cls is not None and not isinstance(cls, type) and name is None:
        assert isinstance(cls, str), "name of the model must be a string"
        cls, name = None, cls

    assert name is not None, "name of the model is required"
    if cls is not None:
        saveto_registry[name] = cls
        return saveto_registry[name]

    return lambda x: register(x, name, saveto=saveto_registry)
