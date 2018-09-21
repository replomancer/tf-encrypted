import functools
from typing import Optional, Type, List, Dict, Any, Union
from types import TracebackType

import tensorflow as tf
from ..io import InputProvider

from ..tensor.tensor import AbstractTensor
from .pond import PondPrivateTensor, PondMaskedTensor


_current_prot = None
global_cache_updators: List = []
nodes: Dict = {}


class Protocol(object):

    def __enter__(self) -> 'Protocol':
        set_protocol(self)
        return self

    def __exit__(self, type: Optional[Type[BaseException]],
                 value: Optional[Exception],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        set_protocol(None)
        return None

    def define_private_input(
        self,
        provider: InputProvider,
        apply_scaling: bool=True,
        name: Optional[str]=None,
        masked: bool=False
    ) -> Union[PondPrivateTensor, PondMaskedTensor, List[PondPrivateTensor], List[PondMaskedTensor]]:
        raise NotImplementedError()


def set_protocol(prot: Optional[Protocol]) -> None:
    global _current_prot
    _current_prot = prot


def get_protocol() -> Optional[Protocol]:
    return _current_prot


def global_caches_updator():
    with tf.name_scope('cache_update'):
        return tf.group(*global_cache_updators)


def memoize(func):

    @functools.wraps(func)
    def cache_nodes(self: Protocol, *args: Any, **kwargs: Any) -> AbstractTensor:
        args = tuple(tuple(x) if isinstance(x, list) else x for x in args)
        node_key = (func.__name__, args, tuple(sorted(kwargs.items())))

        cached_result = nodes.get(node_key, None)
        if cached_result is not None:
            return cached_result

        result = func(self, *args, **kwargs)

        nodes[node_key] = result
        return result

    return cache_nodes
