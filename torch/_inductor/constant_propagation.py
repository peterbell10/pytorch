import math
import builtins
import operator
import functools
from typing import Union
from dataclasses import dataclass

import torch
from .virtualized import WrapperHandler

def nan_max(a, b):
    return a if math.isnan(a) else max(a, b)

def nan_min(a, b):
    return a if math.isnan(a) else min(a, b)

def sigmoid(x):
    ex = math.exp(-abs(x))
    t = 1 if x > 0 else ex
    return t / (1 + ex)

unary_mappings = {
    'reciprocal': lambda x: 1 / x,
    'abs': builtins.abs,
    'exp': math.exp,
    'exp2': lambda x: torch.tensor(x, torch.double).exp2().item(),
    'expm1': math.expm1,
    'sqrt': math.sqrt,
    'relu': lambda x: nan_max(0, x),
    'cos': math.cos,
    'sin': math.sin,
    'tan': math.tan,
    'cosh': math.cosh,
    'sinh': math.sinh,
    'tanh': math.tanh,
    'erf': math.erf,
    'lgamma': math.lgamma,
    'rsqrt': lambda x: 1 / math.sqrt(x),
    'log': math.log,
    'log1p': math.log1p,
    'sigmoid': sigmoid,
    'signbit': lambda x: '-' in repr(x),
    'isinf': math.isinf,
    'isnan': math.isnan,
    'round': builtins.round,
    'floor': math.floor,
    'trunc': math.trunc,
    'ceil': math.ceil,
    'identity': lambda x: x,
    'square': lambda x: x * x,
    'bitwise_not': lambda x: ~x,
    'logical_not': lambda x: not bool(x),
}

libdevice_unary_functions = [
    'abs',
    'exp',
    'sqrt',
    'cos',
    'sin',
    'tan',
    'sigmoid',
    'log',
]
for name in libdevice_unary_functions:
    unary_mappings['libdevice_' + name] = unary_mappings[name]

binary_mappings = {
    'bitwise_and': operator.and_,
    'bitwise_or': operator.or_,
    'bitwise_xor': operator.xor,
    'logical_and': lambda x, y: bool(x) and bool(y),
    'logical_or': lambda x, y: bool(x) or bool(y),
    'logical_xor': lambda x, y: bool(x) != bool(y),
    'minimum': nan_min,
    'maximum': nan_max,
    'fmod': math.fmod,
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'floordiv': operator.floordiv,
    'div': operator.truediv,
    'mod': operator.mod,
    'pow': builtins.pow,
    'eq': operator.eq,
    'ne': operator.ne,
    'lt': operator.lt,
    'gt': operator.gt,
    'le': operator.le,
    'ge': operator.ge,
}

def is_constant(x):
    return (
        isinstance(x, torch.fx.Proxy) and
        x.node.op == "call_method" and
        x.node.target == "constant"
    )

@dataclass
class ConstantVar:
    value: Union[bool, int, float]
    dtype: torch.dtype

    def __post_init__(self):
        type_ = torch._prims_common.dtype_to_type(self.dtype)
        self.value = type_(self.value)

    @staticmethod
    def from_node(node: torch.fx.Node):
        assert node.target == "constant"
        def inner_fn(self, value, dtype):
            return ConstantVar(value, dtype)
        return inner_fn(*node.args, *node.kwargs)

    @staticmethod
    def from_proxy(proxy: torch.fx.Proxy):
        return ConstantVar.from_node(proxy.node)

def reorder_symmetric(x, y):
    return (y, x) if is_constant(y) else (x, y)


class PropagateConstants(WrapperHandler):
    def _call_unwrapped(self, fn_name, *args, **kwargs):
        return getattr(self._inner, fn_name)(*args, **kwargs)

    def __getattr__(self, item):
        return functools.partial(self._call_unwrapped, item)

    def to_dtype(self, x, dtype):
        if is_constant(x):
            value = ConstantVar.from_proxy(x).value
            return self.constant(value, dtype)
        return self._call_unwrapped('to_dtype', x, dtype)

    def where(self, a, b, c):
        if is_constant(a):
            value = ConstantVar.from_proxy(a).value
            return b if bool(value) else c
        return self._call_unwrapped('where', a, b, c)

    def masked(self, mask, body, other):
        # TODO: cannot determine dtype
        # if is_constant(mask):
        #     value = ConstantVar.from_proxy(mask).value
        #     return body if value else self.constant(other, ...)
        return self._call_unwrapped('masked', mask, body, other)

    def logical_and(self, x, y):
        x, y = reorder_symmetric(x, y)
        if is_constant(x):
            value = ConstantVar.from_proxy(x).value
            return y if bool(value) else self.constant(False, torch.bool)

        return self._call_unwrapped('logical_and', x, y)

    def logical_or(self, x, y):
        x, y = reorder_symmetric(x, y)
        if is_constant(x):
            value = ConstantVar.from_proxy(x).value
            return y if not bool(value) else self.constant(True, torch.bool)

        return self._call_unwrapped('logical_or', x, y)

    def logical_xor(self, x, y):
        x, y = reorder_symmetric(x, y)
        if is_constant(x):
            value = ConstantVar.from_proxy(x).value
            return y if not value else self.logical_not(y)

        return self._call_unwrapped('logical_xor', x, y)

    def bitwise_and(self, x, y):
        x_, y_ = reorder_symmetric(x, y)
        if is_constant(x_):
            x_value, dtype = ConstantVar.from_proxy(x_)
            if x_value == 0:
                return self.constant(0, dtype)
            if is_constant(y_):
                y_value = ConstantVar.from_proxy(y_).value
                return self.constant(x_value & y_value, dtype)

        return self._call_unwrapped('bitwise_and', x, y)

    def bitwise_or(self, x, y):
        x_, y_ = reorder_symmetric(x, y)
        if is_constant(x_):
            x_value, dtype = ConstantVar.from_proxy(x_)
            if x_value == 0:
                return y_
            if is_constant(y):
                y_value = ConstantVar.from_proxy(y_).value
                return self.constant(x_value | y_value, dtype)

        return self._call_unwrapped('bitwise_or', x, y)

    def bitwise_xor(self, x, y):
        x_, y_ = reorder_symmetric(x, y)
        if is_constant(x_):
            x_value, dtype = ConstantVar.from_proxy(x_)
            if x_.value == 0:
                return y_
            if is_constant(y_):
                y_value = ConstantVar.from_proxy(y_).value
                return self.constant(x_value ^ y_value, dtype)

        return self._call_unwrapped('bitwise_xor', x, y)

    def add(self, x, y):
        x_, y_ = reorder_symmetric(x, y)
        if is_constant(x_):
            xc = ConstantVar.from_proxy(x_)
            if xc.value == 0:
                return y
            if is_constant(y_):
                y_value = ConstantVar.from_proxy(y_).value
                return self.constant(xc.value + y_value, xc.dtype)

        return self._call_unwrapped('add', x, y)

    def sub(self, x, y):
        if is_constant(x) and ConstantVar.from_proxy(x).value == 0:
            return self.neg(y)
        if is_constant(y) and y.value == 0:
            return x
        if is_constant(x) and is_constant(y):
            xc = ConstantVar.from_proxy(x)
            yc = ConstantVar.from_proxy(y)
            return self.constant(xc.value - yc.value, xc.dtype)
        return self._call_unwrapped('sub', x, y)


def safe_call_mapping(mapping, *args):
    try:
        return mapping(*args)
    except ValueError:
        # Domain errors
        return None

for name, mapping in unary_mappings.items():
    def make_unary_func(name, mapping):
        def inner_fn(self, x):
            if is_constant(x):
                x = ConstantVar.from_proxy(x)
                new_value = safe_call_mapping(mapping, x.value)
                if new_value is not None:
                    return self.constant(new_value, x.dtype)

            return self._call_unwrapped(name, x)
        return inner_fn

    if not hasattr(PropagateConstants, name):
        setattr(PropagateConstants, name, make_unary_func(name, mapping))

for name, mapping in binary_mappings.items():
    def make_binary_func(name, mapping):
        def inner_fn(self, x, y):
            if is_constant(x) and is_constant(y):
                new_value = safe_call_mapping(mapping, x.value, y.value)
                if new_value is not None:
                    return self._inner.constant(new_value, x.dtype)
            return self._call_unwrapped(name, x, y)
        return inner_fn

    if not hasattr(PropagateConstants, name):
        setattr(PropagateConstants, name, make_binary_func(name, mapping))
