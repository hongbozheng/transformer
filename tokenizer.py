from torch import Tensor

import torch
from notation import CONSTANTS, VARIABLES, OPERATORS, SYMPY_OPERATORS


class Tokenizer:
    def __init__(self) -> None:
        """Tokenize Components in notation.py"""
        self.soe = "SOE"
        self.eoe = "EOE"
        self.pad = "PAD"
        self.constants = CONSTANTS
        self.variables = VARIABLES
        self.operators = OPERATORS
        self.sympy_operators = SYMPY_OPERATORS

        self.symbols = [self.pad, self.soe]
        self.symbols.extend(self.constants)
        self.symbols.extend(list(self.variables.keys()))
        self.symbols.extend(list(self.operators.keys()))
        self.symbols.append(self.eoe)

        self.sym2idx = {sym: i for i, sym in enumerate(self.symbols)}
        self.idx2sym = {i: sym for i, sym in enumerate(self.symbols)}

    def encode(self, expr: str) -> Tensor:
        tokens = []
        for comp in expr.split(sep=' '):
            tokens.append(self.sym2idx[comp])

        tokens = torch.cat(
            tensors=(
                torch.tensor(data=[self.sym2idx["SOE"]], dtype=torch.int64),
                torch.tensor(data=tokens, dtype=torch.int64),
                torch.tensor(data=[self.sym2idx["EOE"]], dtype=torch.int64),
            ),
            dim=0,
        )
        return tokens

    def decode(self, tokens: Tensor) -> str:
        expr = []
        for token in tokens:
            expr.append(self.idx2sym[token.item()])
            if token == self.sym2idx["EOE"]:
                break
        expr = " ".join(expr)
        return expr
