from typing import Dict, List, Tuple

import sympy as sp
from components import CONSTANTS, VARIABLES, OPERATORS, COEFFICIENTS
from sympy import Expr


def write_infix(token: str, args: List) -> str:
    """
    Infix representation.
    Convert prefix expressions to a format that SymPy can parse.
    """
    if token == 'add':
        return f'({args[0]})+({args[1]})'
    elif token == 'sub' or token == 'subtract':
        return f'({args[0]})-({args[1]})'
    elif token == 'mul' or token == 'multiply':
        return f'({args[0]})*({args[1]})'
    elif token == 'div':
        return f'({args[0]})/({args[1]})'
    elif token == 'pow':
        return f'({args[0]})**({args[1]})'
    elif token == 'rac':
        return f'({args[0]})**(1/({args[1]}))'
    elif token == 'and':
        return f'({args[0]})&({args[1]})'
    elif token == 'or':
        return f'({args[0]})|({args[1]})'
    elif token == 'xor':
        return f'({args[0]})^({args[1]})'
    elif token == 'implies':
        return f'({args[0]})>>({args[1]})'
    elif token == 'not':
        return f'~({args[0]})'
    elif token == 'abs':
        return f'Abs({args[0]})'
    elif token == 'inv':
        return f'1/({args[0]})'
    elif token == 'pow2':
        return f'({args[0]})**2'
    elif token == 'pow3':
        return f'({args[0]})**3'
    elif token == 'pow4':
        return f'({args[0]})**4'
    elif token == 'pow5':
        return f'({args[0]})**5'
    elif token in ['sign', 'sqrt', 'exp', 'ln',
                   'sin', 'cos', 'tan',
                   'csc', 'sec', 'cot',
                   'sinh', 'cosh', 'tanh',
                   'csch', 'sech', 'coth',
                   'asin', 'acos', 'atan',
                   'acsc', 'asec', 'acot',
                   'asinh', 'acosh', 'atanh',
                   'acsch', 'asech', 'acoth']:
        return f'{token}({args[0]})'
    elif token == 'd':
        return f'Derivative({args[1]},{args[0]})'
    elif token == 'f':
        return f'f({args[0]})'
    elif token == 'g':
        return f'g({args[0]},{args[1]})'
    elif token == 'h':
        return f'h({args[0]},{args[1]},{args[2]})'
    elif token.startswith('INT'):
        return f'{token[-1]}{args[0]}'
    else:
        return token


def parse_int(lst: List) -> Tuple[int, int]:
    """
    Parse a list that starts with an integer.
    Return the integer value, and the position it ends in the list.
    """
    base = 10
    balanced = False
    val = 0
    # if first token is INT+ or INT-
    if not (balanced and lst[0] == 'INT' or base >= 2 and
            lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
        raise Exception(f"Invalid integer in prefix expression")
    i = 0
    for x in lst[1:]:
        # if the rest part of the list is not a number, break
        if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
            break
        # otherwise, convert the str into int
        val = val * base + int(x)
        i += 1
    if base > 0 and lst[0] == 'INT-':
        val = -val
    # i+1 is the position number ends in the list
    return val, i + 1


def prefix_to_infix(tokens: List[str]) -> Tuple[str, List[str]]:
    """
    Parse an expression in prefix mode, and output it in either:
        - infix mode (returns human readable string)
        - develop mode (returns a dictionary with the simplified expression)
    """
    if len(tokens) == 0:
        raise Exception("Empty prefix list.")
    op = tokens[0]

    # OPERATOR dict, t is an operator
    if op in OPERATORS:
        args = []
        l1 = tokens[1:]
        for _ in range(OPERATORS[op]):
            i1, l1 = prefix_to_infix(l1)
            args.append(i1)
        return write_infix(op, args), l1
    # if t is variable 'x' or coefficient 'a1', 'a2'... ,
    # or constant "pi", "E", or 'I'
    elif op in CONSTANTS or op in VARIABLES or op in COEFFICIENTS or op == 'I':
        return op, tokens[1:]
    # else when op is INT+ INT-
    else:
        val, i = parse_int(tokens)
        return str(val), tokens[i:]


def get_sympy_local_dict() -> Dict:
    local_dict = {}
    for k, v in list(VARIABLES.items()) + list(COEFFICIENTS.items()):
        assert k not in local_dict
        local_dict[k] = v
    return local_dict


def prefix_to_sympy(expr: str, evaluate: bool = True) -> Expr:
    tokens = expr.split(sep=" ")
    p, r = prefix_to_infix(tokens=tokens)
    if len(r) > 0:
        raise Exception(
            f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed."
        )

    local_dict = get_sympy_local_dict()
    expr = sp.parsing.sympy_parser.parse_expr(
        s=f'({p})',
        evaluate=evaluate,
        local_dict=local_dict,
    )
    return expr
