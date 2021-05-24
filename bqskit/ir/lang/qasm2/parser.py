"""This module defines the OPENQASM 2.0 Grammer and Lark Parser for it."""
from __future__ import annotations

from lark import Lark
from lark import Tree

_OPENQASMPARSER = Lark(
    r"""
ID: /[a-z][A-Za-z0-9_]*/
REAL: /([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?/
NNINTEGER: /[1-9]+[0-9]*|0/
PI: "pi"
SIN: "sin"
COS: "cos"
TAN: "tan"
EXP: "EXP"
LN: "ln"
SQRT: "sqrt"
mainprogram: "OPENQASM" REAL ";" program
program: statement | program statement
statement: decl
            | gatedecl goplist "}"
            | gatedecl "}"
            | "opaque" ID idlist ";"
            | "opaque" ID "( )" idlist ";"
            | "opaque" ID "(" idlist ")" idlist ";"
            | qop
            | "if (" ID "==" NNINTEGER ")" qop
            | "barrier" anylist ";"
            | "include" ESCAPED_STRING ";"
            | /\/\/+.*/
decl: qreg | creg
creg: "creg" ID "[" NNINTEGER "]" ";"
qreg: "qreg" ID "[" NNINTEGER "]" ";"
gatedecl: "gate" ID idlist "{"
            | "gate" ID "( )" idlist "{"
            | "gate" ID "(" idlist ")" idlist "{"
goplist: uop
            | "barrier" idlist ";"
            | goplist uop
            | goplist "barrier" idlist ";"
qop: uop
        | measure
        | reset
measure: "measure" argument "->" argument ";"
reset: "reset" argument ";"
uop: ugate
        | cxgate
        | gate
gate: ID anylist ";"
        | ID "( )" anylist ";"
        | ID "(" explist ")" anylist ";"
ugate: "U (" explist ")" argument ";"
cxgate: "CX" argument "," argument ";"
anylist: idlist | mixedlist
idlist: ID | idlist "," ID
mixedlist: ID "[" NNINTEGER "]" | mixedlist "," ID
                                    | mixedlist "," ID "[" NNINTEGER "]"
                                    | idlist "," ID "[" NNINTEGER "]"
argument: ID | ID "[" NNINTEGER "]"
explist: exp | explist "," exp
exp: mulexp ((/\+/ | /\-/) mulexp)*
mulexp: primaryexp ((/\*/ | /\//) primaryexp)*
usub: "-" exp
pow: primaryexp "^" primaryexp
parenexp: "(" exp ")"
unaryexp: unaryop "(" exp ")"
primaryexp: parenexp | REAL | NNINTEGER | PI | ID | pow | usub | unaryexp
unaryop: SIN
        | COS
        | TAN
        | EXP
        | LN
        | SQRT

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
""",
    parser='lalr',
    start='mainprogram',
)


def parse(s: str) -> Tree:
    return _OPENQASMPARSER.parse(s)
