from __future__ import annotations

from typing import Any


class Dict(dict):

    def __getattr__(self, name: str) -> Any:
        return dict.__getitem__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        return dict.__setitem__(self, name, value)

    def __delattr__(self, name: str) -> None:
        return dict.__delitem__(self, name)

    def __repr__(self) -> str:
        """Representation of the Dict object"""
        keys = list(self.keys())
        if keys:
            m = max(map(len, keys)) + 1
            return "\n".join([k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())])
        return self.__class__.__name__ + "()"
