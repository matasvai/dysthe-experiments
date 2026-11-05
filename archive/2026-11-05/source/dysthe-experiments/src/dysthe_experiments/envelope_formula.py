"""Bounded LaTeX subset for initial envelopes; no general-purpose execution.

This module parses its own grammar and evaluates only explicit NumPy operations.
It does not establish resolution or physical validity of an initial condition.
"""
from dataclasses import dataclass
import math
import re

import numpy as np


PARSER_VERSION = 'dysthe-envelope-latex-v1'
FORMULA_DEFAULT = r'A \exp(-\frac{\tau^2}{2 w^2}) \exp(i(k\tau+c\tau^2))'
SYMBOL_DEFAULTS = dict(A=1., w=2., k=.3, c=0.)
MAX_LENGTH = 1024
MAX_TOKENS = 512
MAX_NODES = 256
MAX_DEPTH = 32
MAX_SYMBOLS = 8
MAX_SAMPLES = 16384
MAX_POWER = 12
MAX_LITERAL_LENGTH = 32
MAX_INPUT_MAGNITUDE = 1e6
MAX_INTERMEDIATE_MAGNITUDE = 1e12
_NUMBER = re.compile(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', re.ASCII)
_WORD = re.compile(r'[A-Za-z]+', re.ASCII)
_FUNCTIONS = {'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
              'sech': lambda x: np.divide(1., np.cosh(x))}
_BUILTINS = ('sech', 'exp', 'sin', 'cos', 'tau', 'pi')


class FormulaError(ValueError):
    """A syntax, declaration, input-shape, or bounded-evaluation error."""


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    position: int


@dataclass(frozen=True)
class _Node:
    kind: str
    value: object = None
    children: tuple = ()
    height: int = 1


def _symbols(value):
    if not isinstance(value, dict):
        raise FormulaError('Symbols must be a JSON object of declared real scalar values.')
    if len(value) > MAX_SYMBOLS:
        raise FormulaError(f'At most {MAX_SYMBOLS} scalar symbols may be declared.')
    result = []
    for name, scalar in value.items():
        if (type(name) is not str or len(name) != 1 or not name.isascii()
                or not name.isalpha() or name == 'i'):
            raise FormulaError('Symbol names must be single ASCII letters; i is reserved.')
        if type(scalar) not in (int, float):
            raise FormulaError(f'Symbol {name} must be a finite real number, not a boolean or array.')
        if abs(scalar) > MAX_INPUT_MAGNITUDE or not math.isfinite(scalar):
            raise FormulaError(f'Symbol {name} must be finite with magnitude <= 1e6.')
        result.append((name, float(scalar)))
    return tuple(sorted(result))


def _tokenize(formula):
    if type(formula) is not str or not formula.strip():
        raise FormulaError('Enter a nonempty formula for the right-hand side of u(0, tau).')
    if len(formula) > MAX_LENGTH:
        raise FormulaError(f'Formula is too long; the limit is {MAX_LENGTH} characters.')
    tokens = []
    cursor = 0
    while cursor < len(formula):
        char = formula[cursor]
        if char.isspace():
            cursor += 1
            continue
        start = cursor
        if char in '+-*/^(){}':
            tokens.append(_Token(char, char, start))
            cursor += 1
        elif char == '\\':
            match = _WORD.match(formula, cursor+1)
            if match is None:
                raise FormulaError(f'Unsupported LaTeX command at character {start+1}.')
            name = match.group()
            cursor = match.end()
            if name in ('left', 'right'):
                while cursor < len(formula) and formula[cursor].isspace():
                    cursor += 1
                expected = '(' if name == 'left' else ')'
                if cursor == len(formula) or formula[cursor] != expected:
                    raise FormulaError(f'Use \\{name}{expected}; only round scalable delimiters are supported.')
                tokens.append(_Token(expected, expected, start))
                cursor += 1
            elif name in ('cdot', 'times'):
                tokens.append(_Token('*', '*', start))
            elif name in _BUILTINS or name == 'frac':
                tokens.append(_Token('name', name, start))
            else:
                raise FormulaError(f'Unsupported command \\{name} at character {start+1}.')
        elif char.isascii() and (char.isdigit() or char == '.'):
            match = _NUMBER.match(formula, cursor)
            if match is None:
                raise FormulaError(f'Invalid number at character {start+1}.')
            literal = match.group()
            if len(literal) > MAX_LITERAL_LENGTH:
                raise FormulaError(f'Numeric literals are limited to {MAX_LITERAL_LENGTH} characters.')
            number = float(literal)
            if not math.isfinite(number) or number > MAX_INPUT_MAGNITUDE:
                raise FormulaError('Numeric literals must be finite with magnitude <= 1e6.')
            tokens.append(_Token('number', literal, start))
            cursor = match.end()
        elif char.isascii() and char.isalpha():
            name = next((word for word in _BUILTINS if formula.startswith(word, cursor)), char)
            tokens.append(_Token('name', name, start))
            cursor += len(name)
        else:
            raise FormulaError(f'Unsupported character {char!r} at character {start+1}.')
        if len(tokens) > MAX_TOKENS:
            raise FormulaError(f'Formula has too many tokens; the limit is {MAX_TOKENS}.')
    tokens.append(_Token('end', '', len(formula)))
    return tokens


class _Parser:
    def __init__(self, tokens, symbols):
        self.tokens = tokens
        self.declared = set(symbols)
        self.cursor = 0
        self.nodes = 0

    @property
    def token(self):
        return self.tokens[self.cursor]

    def take(self, kind):
        token = self.token
        if token.kind != kind:
            label = 'the end of the formula' if token.kind == 'end' else repr(token.text)
            raise FormulaError(f'Expected {kind!r}, found {label} at character {token.position+1}.')
        self.cursor += 1
        return token

    def node(self, kind, value=None, children=()):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise FormulaError(f'Formula is too complex; the limit is {MAX_NODES} expression nodes.')
        height = 1 + max((child.height for child in children), default=0)
        self.depth(height)
        return _Node(kind, value, children, height)

    @staticmethod
    def depth(value):
        if value > MAX_DEPTH:
            raise FormulaError(f'Formula nesting is too deep; the limit is {MAX_DEPTH}.')

    def parse(self):
        result = self.sum(0)
        self.take('end')
        return result

    def sum(self, depth):
        self.depth(depth)
        result = self.product(depth)
        while self.token.kind in ('+', '-'):
            operation = self.token.kind
            self.cursor += 1
            result = self.node(operation, children=(result, self.product(depth)))
        return result

    def product(self, depth):
        result = self.unary(depth)
        while self.token.kind in ('*', '/') or self.token.kind in ('number', 'name', '(', '{'):
            if self.token.kind in ('*', '/'):
                operation = self.token.kind
                self.cursor += 1
            else:
                operation = '*'
            result = self.node(operation, children=(result, self.unary(depth)))
        return result

    def unary(self, depth):
        self.depth(depth)
        if self.token.kind in ('+', '-'):
            operation = self.token.kind
            self.cursor += 1
            return self.node('unary'+operation, children=(self.unary(depth+1),))
        return self.power(depth)

    def power(self, depth):
        result = self.atom(depth)
        if self.token.kind == '^':
            self.cursor += 1
            opening = self.token.kind if self.token.kind in ('(', '{') else None
            if opening:
                self.cursor += 1
            sign = 1
            if self.token.kind in ('+', '-'):
                sign = -1 if self.token.kind == '-' else 1
                self.cursor += 1
            literal = self.take('number').text
            if not literal.isascii() or not literal.isdigit() or len(literal) > 2:
                raise FormulaError('Powers require a literal integer exponent from -12 through 12.')
            exponent = sign*int(literal)
            if abs(exponent) > MAX_POWER:
                raise FormulaError('Powers require a literal integer exponent from -12 through 12.')
            if opening:
                self.take(')' if opening == '(' else '}')
            result = self.node('power', exponent, (result,))
        return result

    def group(self, depth, required=None):
        self.depth(depth)
        opening = self.token.kind
        if opening not in ('(', '{') or required is not None and opening != required:
            raise FormulaError(f'Expected a {"braced" if required == "{" else "parenthesized or braced"} argument at character {self.token.position+1}.')
        self.cursor += 1
        result = self.sum(depth+1)
        self.take(')' if opening == '(' else '}')
        return result

    def atom(self, depth):
        token = self.token
        if token.kind in ('(', '{'):
            return self.group(depth)
        if token.kind == 'number':
            self.cursor += 1
            return self.node('number', float(token.text))
        if token.kind == 'name':
            self.cursor += 1
            name = token.text
            if name == 'frac':
                return self.node('/', children=(self.group(depth, '{'), self.group(depth, '{')))
            if name in _FUNCTIONS:
                return self.node('function', name, (self.group(depth),))
            if name in ('tau', 'i', 'pi'):
                return self.node(name)
            if name in self.declared:
                return self.node('symbol', name)
            raise FormulaError(f'Undeclared symbol {name!r} at character {token.position+1}; declare its real scalar value.')
        raise FormulaError(f'Expected a number, declared symbol, or supported function at character {token.position+1}.')


def _render(node):
    if node.kind == 'number':
        return repr(node.value)
    if node.kind in ('tau', 'i', 'pi'):
        return node.kind
    if node.kind == 'symbol':
        return node.value
    if node.kind.startswith('unary'):
        return '('+node.kind[-1]+_render(node.children[0])+')'
    if node.kind == 'function':
        return node.value+'('+_render(node.children[0])+')'
    if node.kind == 'power':
        return '('+_render(node.children[0])+' ^ '+str(node.value)+')'
    return '('+_render(node.children[0])+' '+node.kind+' '+_render(node.children[1])+')'


@dataclass(frozen=True)
class CompiledFormula:
    formula: str
    _symbols: tuple
    _root: _Node
    node_count: int

    def provenance(self):
        """Return an independent JSON-safe description of the sampled prescription."""
        return dict(parser_version=PARSER_VERSION, formula=self.formula,
                    symbols=dict(self._symbols), normalized_expression=_render(self._root),
                    expression_nodes=self.node_count,
                    limits=dict(characters=MAX_LENGTH, tokens=MAX_TOKENS,
                        expression_nodes=MAX_NODES, nesting=MAX_DEPTH,
                        symbols=MAX_SYMBOLS, samples=MAX_SAMPLES,
                        absolute_integer_power=MAX_POWER,
                        input_magnitude=MAX_INPUT_MAGNITUDE,
                        intermediate_magnitude=MAX_INTERMEDIATE_MAGNITUDE))

    def evaluate(self, tau):
        """Evaluate over one finite real tau axis, returning copied complex128 data."""
        try:
            axis = np.asarray(tau)
        except (TypeError, ValueError) as error:
            raise FormulaError('tau must be a one-dimensional real numeric array.') from error
        if (axis.ndim != 1 or not 1 <= axis.size <= MAX_SAMPLES
                or axis.dtype.kind not in 'iuf'):
            raise FormulaError(f'tau must be a one-dimensional real numeric array with 1 through {MAX_SAMPLES} samples.')
        if (not np.isfinite(axis).all() or np.any(axis < -MAX_INPUT_MAGNITUDE)
                or np.any(axis > MAX_INPUT_MAGNITUDE)):
            raise FormulaError('tau samples must be finite with magnitude <= 1e6.')
        axis = axis.astype(np.float64, copy=False)
        symbols = dict(self._symbols)

        def checked(value):
            value = np.asarray(value, dtype=np.complex128)
            if value.shape not in ((), axis.shape):
                raise FormulaError('Expression output cannot broadcast to the tau axis.')
            if not np.isfinite(value).all() or np.any(np.abs(value) > MAX_INTERMEDIATE_MAGNITUDE):
                raise FormulaError('Expression intermediate values must be finite with magnitude <= 1e12; reduce parameters or revise the formula.')
            return value

        def visit(node):
            kind = node.kind
            if kind == 'tau':
                return checked(axis)
            if kind == 'i':
                return checked(1j)
            if kind == 'pi':
                return checked(np.pi)
            if kind == 'number':
                return checked(node.value)
            if kind == 'symbol':
                return checked(symbols[node.value])
            left = visit(node.children[0])
            if kind == 'function':
                value = _FUNCTIONS[node.value](left)
            elif kind == 'power':
                value = np.power(left, node.value)
            elif kind == 'unary+':
                value = left
            elif kind == 'unary-':
                value = np.negative(left)
            else:
                right = visit(node.children[1])
                value = {'+': np.add, '-': np.subtract, '*': np.multiply,
                         '/': np.divide}[kind](left, right)
            return checked(value)

        try:
            with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
                result = visit(self._root)
                return np.broadcast_to(result, axis.shape).astype(np.complex128, copy=True)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise FormulaError('Expression overflowed, divided by zero, or produced a non-finite value; reduce parameters or revise the formula.') from error


def compile_formula(formula, symbols):
    """Validate syntax/declarations and return a reusable bounded evaluator."""
    declared = _symbols(symbols)
    parser = _Parser(_tokenize(formula), dict(declared))
    root = parser.parse()
    return CompiledFormula(formula, declared, root, parser.nodes)


def validate_formula(formula, symbols):
    """Alias for compile_formula; sampling is still required to validate values."""
    return compile_formula(formula, symbols)
