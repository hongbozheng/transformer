import sympy as sp
from collections import OrderedDict
import torch
from torch import Tensor


OPERATORS = {
    # Elementary functions
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
    # Trigonometric Functions
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "cot": 1,
    "sec": 1,
    "csc": 1,
    # Trigonometric Inverses
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "acot": 1,
    "asec": 1,
    "acsc": 1,
    # Hyperbolic Functions
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "coth": 1,
    "sech": 1,
    "csch": 1,
    # Hyperbolic Inverses
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "acoth": 1,
    "asech": 1,
    "acsch": 1,
}
CONSTANTS = ["pi", "E"]
VARIABLES = OrderedDict({
    "x": sp.Symbol("x", real=True, nonzero=True),
})
SYMBOLS = ["I", "INT+", "INT-", "INT", "FLOAT", "-", ".", "10^", "Y"]
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
SPECIAL_WORDS = ["SOE", "EOE", "PAD"]
INT_BASE = 10
COEFFICIENTS = OrderedDict({
    f'a{i}': sp.Symbol(f'a{i}', real=True)
    for i in range(10)
})
elements = [str(i) for i in range(abs(INT_BASE))]

##################################################################################

filepath = 'C:/Users/wsylxy/Desktop/UIUC/project/expemb/test.txt'
max_seq_len = 512
file = open(filepath, "r", encoding="utf-8")
skipped = 0
for idx, line in enumerate(file):
    print(line)
    eq_tuples = []
    line = line.strip() #remove space character at the beginning and the end of string
    if "\t" in line:
        ip_eq = line.split("\t")[0]
        op_eq = line.split("\t")[1]
        print(ip_eq)
        print(op_eq)
    else:
        ip_eq = line
        op_eq = line

    ip_len = len(ip_eq.split(" "))
    op_len = len(op_eq.split(" "))
    if max_seq_len == -1:
        eq_tuples.append((ip_eq, op_eq))
    elif ip_len <= max_seq_len and op_len <= max_seq_len:
        eq_tuples.append((ip_eq, op_eq))
    else:
        skipped += 1

print(f"Skipped {skipped} lines due to max sequence length restriction.")
file.close()

#################################################################################

components = SPECIAL_WORDS + CONSTANTS + list(VARIABLES.keys()) + list(OPERATORS.keys()) + SYMBOLS + elements
comp2index = {comp : idx for idx, comp in enumerate(components)}
index2comp = {idx : comp for comp, idx in comp2index.items()}


def get_index(comp: str) -> int:
    return comp2index[comp]
print(comp2index)


def encode(exp: str) -> Tensor: #convert token into index of token, add 'SOE' and 'EOE'
    print('encode:', exp)
    # indexes = [self.get_index("SOE")] + [self.get_index(comp) for comp in exp.split(" ")] + [self.get_index("EOE")]
    comp_index = []
    for comp in exp.split(" "):
        print(comp)
        comp_index.append(get_index(comp))
    indexes = [get_index("SOE")] + comp_index + [get_index("EOE")]
    print(indexes)
    return torch.LongTensor(indexes).view(-1)



eq_tuple = eq_tuples[0]
print('encode(eq_tuple[0]):',encode(eq_tuple[0]))
print('encode(eq_tuple[1]):',encode(eq_tuple[1]))