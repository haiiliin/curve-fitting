from __future__ import annotations

import json
from hashlib import sha512
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel
from typing_extensions import Self


class HashableBaseModelIO(BaseModel, extra="forbid", from_attributes=True):
    """Input/output utilities for the models with support for the following features:

    - Hashing of the model
    - Conversion to and from dictionaries, json, toml, and yaml files
    - Compatibility with pydantic v1 and v2
    """

    @property
    def exclude(self) -> set[int] | set[str] | dict[int, Any] | dict[str, Any] | None:
        """Fields to exclude from the model, typically used to exclude arbitrary types when it is allowed in the
        pydantic model to avoid hashing issues."""
        return None

    def __hash__(self) -> int:
        """Return the hash of the model."""
        string = f"{self.__class__.__qualname__}::{self.model_dump_json(exclude=self.exclude)}"
        return int.from_bytes(sha512(string.encode("utf-8", errors="ignore")).digest())

    def model_dump(self, **kwargs) -> dict[str, Any]:
        try:
            return super().model_dump(**kwargs)
        except AttributeError:
            return self.dict(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        try:
            return super().model_dump_json(**kwargs)
        except AttributeError:
            return self.json(**kwargs)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> Self:
        try:
            return super().model_validate(obj, **kwargs)
        except AttributeError:
            return super().parse_obj(obj, **kwargs)

    def update(self, data: dict = None, **kwargs) -> Self:
        """Update the options of the optimizer"""
        data = dict(data or {}, **kwargs)
        update = self.dict()
        update.update(data)
        for k, v in self.validate(update).dict(exclude_defaults=True).items():
            setattr(self, k, v)
        return self

    @classmethod
    def fromDict(cls, data: dict, **kwargs) -> Self:
        """Load the model from a dictionary."""
        return cls(**data, **kwargs)

    def toDict(self, **kwargs) -> dict:
        """Convert the model to a dictionary."""
        try:
            return self.model_dump(**kwargs)
        except AttributeError:
            return self.dict(**kwargs)  # noqa

    @classmethod
    def fromJson(cls, *, string: str | None = None, path: str | None = None, **kwargs):
        """Load the model from a json string or file if a path is provided."""
        assert string is not None or path is not None, "Either a string or a path must be provided."
        if string is None:
            with open(path, "r+", encoding="utf-8") as f:
                string = f.read()
        return cls.fromDict(data=json.loads(string), **kwargs)

    def toJson(self, *, path: Path | str | None = None, **kwargs) -> str | None:
        """Convert the model to a json string or save it to a file if a path is provided."""
        string = json.dumps(self.toDict(**kwargs), indent=4)
        if path is not None:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(string)
        else:
            return string

    @classmethod
    def fromToml(cls, *, string: str | None = None, path: str | None = None, **kwargs):
        """Load the model from a toml string or file if a path is provided."""
        try:
            import tomllib as toml  # Python >= 3.11
        except ImportError:
            import tomli as toml  # noqa Python < 3.11

        assert string is not None or path is not None, "Either a string or a path must be provided."
        if string is None:
            with open(path, "r+", encoding="utf-8") as f:
                string = f.read()
        return cls.fromDict(data=toml.loads(string), **kwargs)

    def toToml(self, *, path: Path | str | None = None, **kwargs) -> str | None:
        """Convert the model to a toml string or save it to a file if a path is provided."""
        import tomli_w as toml

        string = toml.dumps(self.toDict(**kwargs))
        if path is not None:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(string)
        else:
            return string

    @classmethod
    def fromYaml(cls, *, string: str | None = None, path: str | None = None, **kwargs):
        """Load the model from a yaml string or file if a path is provided."""
        import yaml

        assert string is not None or path is not None, "Either a string or a path must be provided."
        if string is None:
            with open(path, "r+", encoding="utf-8") as f:
                string = f.read()
        return cls.fromDict(data=yaml.safe_load(string), **kwargs)

    def toYaml(self, *, path: Path | str | None = None, **kwargs) -> str | None:
        """Convert the model to a yaml string or save it to a file if a path is provided."""
        import yaml

        string = yaml.safe_dump(self.toDict(**kwargs))
        if path is not None:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(string)
        else:
            return string

    @classmethod
    def fromBytes(cls, *, binary: bytes | None = None, path: str | None = None, **kwargs) -> Self:
        """Load the model from a binary string or file if a path is provided."""
        import pickle

        assert binary is not None or path is not None, "Either a binary string or a path must be provided."
        if binary is None:
            with open(path, "rb+") as f:
                binary = f.read()
        return cls.fromDict(data=pickle.loads(binary), **kwargs)

    def toBytes(self, *, path: Path | str | None = None, **kwargs) -> bytes | None:
        """Convert the model to a binary string or save it to a file if a path is provided."""
        import pickle

        binary = pickle.dumps(self.toDict(**kwargs))
        if path is not None:
            with open(path, "wb+") as f:
                f.write(binary)
        else:
            return binary

    @classmethod
    def fromAny(
        cls,
        *,
        data: dict | None = None,
        text: str | bytes | None = None,
        path: str | None = None,
        **kwargs,
    ) -> Self:
        """Load the model from a dictionary, json, toml, yaml, or binary string, or file if a path is provided."""
        if data is not None:
            return cls.fromDict(data=data, **kwargs)
        elif text is not None:
            if isinstance(text, str):
                try:
                    return cls.fromJson(string=text, **kwargs)
                except json.JSONDecodeError:
                    try:
                        return cls.fromToml(string=text, **kwargs)
                    except Exception:  # noqa
                        try:
                            return cls.fromYaml(string=text, **kwargs)
                        except Exception:  # noqa
                            pass
                raise ValueError("Could not decode the text string as json, toml, or yaml.")
            else:
                return cls.fromBytes(binary=text, **kwargs)
        elif path is not None:
            if path.endswith(".json"):
                return cls.fromJson(path=path, **kwargs)
            elif path.endswith(".toml"):
                return cls.fromToml(path=path, **kwargs)
            elif path.endswith((".yml", ".yaml")):
                return cls.fromYaml(path=path, **kwargs)
            elif path.endswith((".pkl", ".pickle")):
                return cls.fromBytes(path=path, **kwargs)
            else:
                raise ValueError("Could not determine the file format from the path.")
        else:
            raise ValueError("Either a dictionary, text string, or path must be provided.")


class OptimizerResult(HashableBaseModelIO, extra="allow"):
    """
    OptimizeResult for the optimizers
    """

    #: The solution of the optimization.
    x: List[float] = []
    #: standard deviations of the parameters
    stds: List[float] = []
    #: Values of objective function
    fun: float = 0.0
    #: Number of evaluations of the objective functions
    nfev: int = 0
    #: Number of iterations performed by the optimizer.
    nit: int = 0
    #: A dict of parameters
    parameters: Dict[str, float] = {}
    #: Optimize keys
    optimizeKeys: List[str] = []
    #: Duration
    duration: float = 0.0

    #: fitness for all experiments and lines
    fitness: Dict[str, Dict[str, float]] = {}
    #: weights for all experiments and lines
    weights: Dict[str, Dict[str, float]] = {}
