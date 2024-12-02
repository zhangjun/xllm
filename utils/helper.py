from __future__ import annotations

import functools
import inspect
import os
from typing import ClassVar, Dict


@functools.lru_cache(maxsize=None)
def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


class ContextVar:
    _cache: ClassVar[Dict[str, ContextVar]] = {}
    value: int
    key: str

    def __new__(cls, key, default_value):
        if key in ContextVar._cache:
            return ContextVar._cache[key]
        instance = ContextVar._cache[key] = super().__new__(cls)
        instance.value, instance.key = getenv(key, default_value), key
        return instance

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x


DEBUG = ContextVar("DEBUG", 0)
HF_MODEL_BASE_DIR = ContextVar("HF_MODEL_BASE_DIR", "")
TEST_CALIB_DATA = ContextVar(
    "TEST_CALIB_DATA", "/share_nfs/c4_dataset/c4-train.00000-of-01024.json.gz"
)


def can_accept_argument(func, arg_name):
    """Check if the function can accept a given keyword argument."""
    sig = inspect.signature(func)
    return arg_name in sig.parameters
