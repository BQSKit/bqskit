"""This module defines the OPENQASM 2.0 Grammer and Lark Parser for it."""
from __future__ import annotations

from lark import Lark
from lark import Tree

_OPENQASMPARSER = Lark(
    r"""
ID: /[a-zA-Z][A-Za-z0-9_]*/
REAL: /([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?/
NNINTEGER: /[1-9]+[0-9]*|0/
PI: "pi"
SIN: "sin"
COS: "cos"
TAN: "tan"
EXP: "EXP"
LN: "ln"
SQRT: "sqrt"
COMMENT: /\/\/+.*/
mainprogram: "OPENQASM" REAL ";" program
program: statement | program statement
statement: decl
            | gatedecl goplist rbracket
            | gatedecl rbracket
            | "opaque" ID idlist ";"
            | "opaque" ID "(" ")" idlist ";"
            | "opaque" ID "(" idlist ")" idlist ";"
            | qop
            | "if" "(" ID "==" NNINTEGER ")" qop
            | barrier
            | incstmt
incstmt: "include" ESCAPED_STRING ";"
decl: qreg | creg
creg: "creg" ID "[" NNINTEGER "]" ";"
qreg: "qreg" ID "[" NNINTEGER "]" ";"
barrier: "barrier" anylist ";"
barrierp: "barrier" anylist ";"
gatedecl: "gate" ID idlist "{"
            | "gate" ID "(" ")" idlist "{"
            | "gate" ID "(" idlist ")" idlist "{"
goplist: uopp
            | "barrierp" idlist ";"
            | goplist uopp
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
        | ID "(" ")" anylist ";"
        | ID "(" explist ")" anylist ";"
ugate: "U" "(" explist ")" argument ";"
cxgate: "CX" argument "," argument ";"
uopp: ugatep
        | cxgatep
        | gatep
gatep: ID anylist ";"
        | ID "(" ")" anylist ";"
        | ID "(" explist ")" anylist ";"
ugatep: "U" "(" explist ")" argument ";"
cxgatep: "CX" argument "," argument ";"
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
rbracket: "}"

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
%ignore COMMENT
""",
    parser='lalr',
    start='mainprogram',
)


def parse(s: str) -> Tree:
    return _OPENQASMPARSER.parse(s)
