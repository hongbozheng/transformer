import sympy as sp
from collections import OrderedDict


CONSTANTS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "pi", "e",
    "INT+", "INT-",
]

VARIABLES = OrderedDict({
    'x': sp.Symbol('x', real=None, nonzero=None, positive=None),
})

OPERATORS = {
    # elementary
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "rac": 2,
    "inv": 1,
    "pow2": 1,
    "pow3": 1,
    "pow4": 1,
    "pow5": 1,
    "sqrt": 1,
    "exp": 1,
    "ln": 1,
    "abs": 1,
    "sign": 1,
    # trig
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "csc": 1,
    "sec": 1,
    "cot": 1,
    # inv trig
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "acsc": 1,
    "asec": 1,
    "acot": 1,
    # hyper
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "csch": 1,
    "sech": 1,
    "coth": 1,
    # inv hyper
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "acsch": 1,
    "asech": 1,
    "acoth": 1,
    # derivative
    "d": 2,
}

SYMPY_OPERATORS = {
    # Elementary functions
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Pow: "pow",
    sp.exp: "exp",
    sp.log: "ln",
    sp.Abs: "abs",
    sp.sign: "sign",
    # Trigonometric Functions
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.cot: "cot",
    sp.sec: "sec",
    sp.csc: "csc",
    # Trigonometric Inverses
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.acot: "acot",
    sp.asec: "asec",
    sp.acsc: "acsc",
    # Hyperbolic Functions
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.coth: "coth",
    sp.sech: "sech",
    sp.csch: "csch",
    # Hyperbolic Inverses
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.acoth: "acoth",
    sp.asech: "asech",
    sp.acsch: "acsch",
}

COEFFICIENTS = OrderedDict({
    f'a{i}': sp.Symbol(f'a{i}', real=True)
    for i in range(10)
})
