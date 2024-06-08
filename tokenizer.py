import torch
from torch import Tensor
from components import CONSTANTS, SPECIAL_WORDS, SYMBOLS, VARIABLES, \
    OPERATORS, SYMPY_OPERATORS


class Tokenizer:
    def __init__(self) -> None:
        """Tokenize Components in components.py"""
        self.special_words = SPECIAL_WORDS
        self.constants = CONSTANTS
        self.symbols = SYMBOLS
        self.variables = VARIABLES
        self.operators = OPERATORS
        self.sympy_operators = SYMPY_OPERATORS

        self.components = []
        self.components.extend(self.special_words)
        self.components.extend(self.constants)
        self.components.extend(self.symbols)
        self.components.extend(list(self.variables.keys()))
        self.components.extend(list(self.operators.keys()))

        self.comp2idx = {comp: i for i, comp in enumerate(self.components)}
        self.idx2comp = {i: comp for i, comp in enumerate(self.components)}
        return

    def encode(self, expr: str) -> Tensor:
        tokens = []
        for comp in expr.split(sep=' '):
            tokens.append(self.comp2idx[comp])

        tokens = torch.cat(
            tensors=(
                torch.tensor(data=[self.comp2idx["SOE"]], dtype=torch.int64),
                torch.tensor(data=tokens, dtype=torch.int64),
                torch.tensor(data=[self.comp2idx["EOE"]], dtype=torch.int64),
            ),
            dim=0,
        )
        return tokens

    def decode(self, tokens: Tensor) -> str:
        expr = []
        for token in tokens:
            expr.append(self.idx2comp[token.item()])
            if token == self.comp2idx["EOE"]:
                break
        expr = " ".join(expr)
        return expr