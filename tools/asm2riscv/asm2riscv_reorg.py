#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import string
import argparse
import itertools
import functools
import struct
import subprocess
import tempfile
import re
import logging
from typing import Any, Dict, List, Type, Tuple, Union, Callable, Iterable, Optional

# Logger will be configured in main()
log = logging.getLogger("asm2asm")

# ============================================================================
# 基础工具类
# ============================================================================

class Token:
    """词法标记"""

    tag: int
    val: Union[int, str]
    TOKEN_END, TOKEN_REG, TOKEN_IMM, TOKEN_NUM, TOKEN_NAME, TOKEN_PUNC = range(6)

    def __init__(self, tag: int, val: Union[int, str]):
        self.val, self.tag = val, tag

    @classmethod
    def end(cls):
        return cls(cls.TOKEN_END, "")

    @classmethod
    def reg(cls, reg: str):
        return cls(cls.TOKEN_REG, reg)

    @classmethod
    def imm(cls, imm: int):
        return cls(cls.TOKEN_IMM, imm)

    @classmethod
    def num(cls, num: int):
        return cls(cls.TOKEN_NUM, num)

    @classmethod
    def name(cls, name: str):
        return cls(cls.TOKEN_NAME, name)

    @classmethod
    def punc(cls, punc: str):
        return cls(cls.TOKEN_PUNC, punc)

    def __repr__(self):
        tags = {
            self.TOKEN_END: "<END>",
            self.TOKEN_REG: "<REG %s>",
            self.TOKEN_IMM: "<IMM %d>",
            self.TOKEN_NUM: "<NUM %d>",
            self.TOKEN_NAME: "<NAME %s>",
            self.TOKEN_PUNC: "<PUNC %s>",
        }
        if self.tag in tags:
            return tags[self.tag] % (
                repr(self.val) if self.tag == self.TOKEN_NAME else self.val
            )
        return f"<UNK:{self.tag} {repr(self.val)}>"


class Expression:
    """表达式解析器"""

    pos: int
    src: str

    def __init__(self, src: str):
        self.pos, self.src = 0, src

    @property
    def _ch(self) -> str:
        return self.src[self.pos]

    @property
    def _eof(self) -> bool:
        return self.pos >= len(self.src)

    def _rch(self) -> str:
        pos, self.pos = self.pos, self.pos + 1
        return self.src[pos]

    def _hex(self, ch: str) -> bool:
        if len(ch) == 1 and ch[0] == "0":
            return self._ch.lower() == "x"
        if len(ch) <= 1 or ch[1].lower() != "x":
            return self._ch.isdigit()
        return self._ch in string.hexdigits

    def _int(self, ch: str) -> Token:
        while not self._eof and self._hex(ch):
            ch += self._rch()
        if ch.lower().startswith("0x"):
            return Token.num(int(ch, 16))
        if ch.startswith("0") and len(ch) > 1:
            return Token.num(int(ch, 8))
        return Token.num(int(ch))

    def _name(self, ch: str) -> Token:
        while not self._eof and (self._ch == "_" or self._ch.isalnum()):
            ch += self._rch()
        return Token.name(ch)

    def _read(self, ch: str) -> Token:
        if ch.isdigit():
            return self._int(ch)
        if ch.isidentifier():
            return self._name(ch)
        if ch in ("*", "<", ">") and not self._eof and self._ch == ch:
            return Token.punc(self._rch() * 2)
        if ch in ("+", "-", "*", "/", "%", "&", "|", "^", "~", "(", ")"):
            return Token.punc(ch)
        raise SyntaxError("invalid character: " + repr(ch))

    def _peek(self) -> Optional[Token]:
        pos = self.pos
        ret = self._next()
        self.pos = pos
        return ret

    def _next(self) -> Optional[Token]:
        while not self._eof and self._ch.isspace():
            self.pos += 1
        return Token.end() if self._eof else self._read(self._rch())

    def _grab(self, tk: Token, getvalue: Callable[[str], int]) -> int:
        if tk.tag == Token.TOKEN_NUM:
            return tk.val
        if tk.tag == Token.TOKEN_NAME:
            return getvalue(tk.val)
        raise SyntaxError("integer or identifier expected, got " + repr(tk))

    __pred__ = [{"<<", ">>"}, {"|"}, {"^"}, {"&"}, {"+", "-"}, {"*", "/", "%"}, {"**"}]
    __binary__ = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a // b,
        "%": lambda a, b: a % b,
        "&": lambda a, b: a & b,
        "^": lambda a, b: a ^ b,
        "|": lambda a, b: a | b,
        "<<": lambda a, b: a << b,
        ">>": lambda a, b: a >> b,
        "**": lambda a, b: a**b,
    }

    def _eval(self, op: str, v1: int, v2: int) -> int:
        return self.__binary__[op](v1, v2)

    def _nest(self, nest: int, getvalue: Callable[[str], int]) -> int:
        ret = self._expr(0, nest + 1, getvalue)
        ntk = self._next()
        if ntk.tag != Token.TOKEN_PUNC or ntk.val != ")":
            raise SyntaxError('")" expected, got ' + repr(ntk))
        return ret

    def _unit(self, nest: int, getvalue: Callable[[str], int]) -> int:
        tk = self._next()
        tt, tv = tk.tag, tk.val
        if tt == Token.TOKEN_NUM:
            return tv
        if tt == Token.TOKEN_NAME:
            return getvalue(tv)
        if tt == Token.TOKEN_PUNC and tv == "(":
            return self._nest(nest, getvalue)
        if tt == Token.TOKEN_PUNC and tv == "+":
            return self._unit(nest, getvalue)
        if tt == Token.TOKEN_PUNC and tv == "-":
            return -self._unit(nest, getvalue)
        if tt == Token.TOKEN_PUNC and tv == "~":
            return ~self._unit(nest, getvalue)
        raise SyntaxError(
            "integer, unary operator or nested expression expected, got " + repr(tk)
        )

    def _term(self, pred: int, nest: int, getvalue: Callable[[str], int]) -> int:
        lv = self._expr(pred + 1, nest, getvalue)
        tk = self._peek()
        while True:
            tt, tv = tk.tag, tk.val
            if tt == Token.TOKEN_END:
                return lv
            if tt != Token.TOKEN_PUNC:
                raise SyntaxError("operator expected, got " + repr(tk))
            if tv not in self.__pred__[pred]:
                return lv
            op = self._next().val
            rv = self._expr(pred + 1, nest, getvalue)
            lv = self._eval(op, lv, rv)
            tk = self._peek()

    def _expr(self, pred: int, nest: int, getvalue: Callable[[str], int]) -> int:
        if pred >= len(self.__pred__):
            return self._unit(nest, getvalue)
        return self._term(pred, nest, getvalue)

    def eval(self, getvalue: Callable[[str], int]) -> int:
        return self._expr(0, 0, getvalue)


class Command:
    """汇编指令命令"""

    cmd: str
    args: List[Union[str, bytes]]

    def __init__(self, cmd: str, args: List[Union[str, bytes]]):
        self.cmd, self.args = cmd, args

    def __repr__(self):
        return f"<CMD {self.cmd} {', '.join(map(repr, self.args))}>"

    @classmethod
    def parse(cls, src: str) -> "Command":
        val = src.split(None, 1)
        cmd = val[0]
        if len(val) == 1:
            return cls(cmd, [])
        
        idx, esc, pos, args, vstr = 0, 0, None, [], val[1]
        ESC_IDLE, ESC_ISTR, ESC_BKSL = 0, 1, 2
        
        while idx < len(vstr):
            nch = vstr[idx]
            idx += 1
            
            if pos is None:
                pos = idx - 1
                
            if nch == "," and esc == ESC_IDLE:
                pos, p = None, pos
                args.append(vstr[p : idx - 1].strip())
            elif nch == '"' and esc == ESC_IDLE:
                esc = ESC_ISTR
            elif nch == '"' and esc == ESC_ISTR:
                esc = ESC_IDLE
                pos, p = None, pos
                args.append(
                    vstr[p:idx].strip()[1:-1].encode("utf-8").decode("unicode_escape")
                )
            elif nch == "\\" and esc == ESC_ISTR:
                esc = ESC_BKSL
            elif esc == ESC_BKSL:
                esc = ESC_ISTR
                
        if pos is not None:
            args.append(vstr[pos:].strip())
        return cls(cmd, args)


class Register:
    """寄存器"""

    reg: str

    def __init__(self, reg: str):
        self.reg = reg.lower()

    def __str__(self):
        return self.reg

    def __repr__(self):
        return f"{{REG {self.reg}}}"


class Parameter:
    """函数参数"""

    name: str
    size: int
    creg: "Register"
    goreg: "Register"

    def __init__(self, name: str, size: int, reg: "Register", goreg: "Register"):
        self.creg, self.goreg, self.name, self.size = reg, goreg, name, size

    def __repr__(self):
        return f"<ARG {self.name}({self.size}): {self.creg}>"


class Prototype:
    """函数原型"""

    args: List[Parameter]
    retv: Optional[Parameter]

    def __init__(self, retv: Optional[Parameter], args: List[Parameter]):
        self.retv, self.args = retv, args

    def __repr__(self):
        if self.retv is None:
            return f"<PROTO ({repr(self.args)})>"
        return f"<PROTO ({repr(self.args)}) -> {repr(self.retv)}>"

    @property
    def argspace(self) -> int:
        return sum(
            [v.size for v in self.args], (0 if self.retv is None else self.retv.size)
        )

    @property
    def inputspace(self) -> int:
        return sum([v.size for v in self.args])


class PrototypeMap(Dict[str, Prototype]):
    """函数原型映射"""

    @staticmethod
    def _align(nb: int) -> int:
        return (((nb - 1) >> 3) + 1) << 3

    @classmethod
    def _retv(cls, ret: str) -> Tuple[str, int, Register, Register]:
        name, size, xmm = cls._args(ret)
        reg = Register("fa0") if xmm else Register("a0")
        return name, size, reg, reg

    @classmethod
    def _args(cls, arg: str, sv: str = "") -> Tuple[str, int, bool]:
        while True:
            if not arg:
                raise SyntaxError("missing type for parameter: " + sv)
            if arg[0] != "_" and not arg[0].isalnum():
                return (sv,) + cls._size(arg.strip())
            if not sv and arg[0].isdigit():
                raise SyntaxError("invalid character: " + repr(arg[0]))
            sv += arg[0]
            arg = arg[1:]

    @classmethod
    def _size(cls, name: str) -> Tuple[int, bool]:
        if name[0] == "*":
            return cls._align(8), False
        if name in ("int8", "uint8", "byte", "bool"):
            return cls._align(1), False
        if name in ("int16", "uint16"):
            return cls._align(2), False
        if name == "float32":
            return cls._align(4), True
        if name in ("int32", "uint32", "rune"):
            return cls._align(4), False
        if name == "float64":
            return cls._align(8), True
        if name in ("int64", "uint64", "uintptr", "int", "Pointer", "unsafe.Pointer"):
            return cls._align(8), False
        raise cls._err(f'unrecognized type "{name}"')

    @classmethod
    def _err(cls, msg: str) -> SyntaxError:
        return SyntaxError(
            msg + ", please keep the companion .go file as simple as possible."
        )

    @classmethod
    def _func(cls, src: List[str], idx: int, depth: int = 0) -> Tuple[str, int]:
        for i in range(idx, len(src)):
            for x in map(lambda c: 1 if c == "(" else -1 if c == ")" else 0, src[i]):
                if depth + x >= 0:
                    depth += x
                else:
                    raise cls._err(f'encountered ")" more than "(" on line {i + 1}')
            if depth == 0:
                return " ".join(src[idx : i + 1]), i + 1
        raise cls._err("unexpected EOF when parsing function signatures")

    @classmethod
    def parse(cls, src: str) -> Tuple[str, "PrototypeMap"]:
        idx, pkg, ret = 0, "", PrototypeMap()
        buf = src.splitlines()
        
        # RISC-V ABI register mapping
        ARGS_ORDER_C = [Register(f"a{i}") for i in range(8)]
        ARGS_ORDER_GO = [Register(f"X{10+i}") for i in range(8)]
        FPARGS_ORDER = [Register(f"fa{i}") for i in range(8)]
        
        while idx < len(buf):
            line = buf[idx].strip()
            if not line:
                idx += 1
                continue
            if line.startswith("package"):
                idx, pkg = idx + 1, line[7:].strip().split()[0]
                log.debug(f"Found package: {pkg}")
                continue
            if line.endswith("{") or not line.startswith("func"):
                idx += 1
                continue
            if line.startswith("type"):
                raise cls._err("type declarations are not supported")
                
            decl, pos = cls._func(buf, idx)
            func, idx = decl[4:].strip(), pos
            nd, pos = 1, func.find("(")
            
            if pos == -1:
                raise cls._err("invalid function prototype: " + decl)
                
            args, name, func = "", func[:pos].strip(), func[pos + 1 :].strip()
            
            if not name or not name.isidentifier():
                continue
                
            while nd and func:
                nch, func = func[0], func[1:]
                nd += 1 if nch == "(" else -1 if nch == ")" else 0
                args += nch
                
            if not nd:
                func = func.strip()
            else:
                raise cls._err("unexpected EOF in prototype: " + decl)
                
            if "," in func:
                raise cls._err("multiple return values are not supported")
                
            if not func:
                retv = None
            elif func.startswith("(") and func.endswith(")"):
                retv = Parameter(*cls._retv(func[1:-1]))
            else:
                raise SyntaxError("badly formatted return argument: " + func)
                
            if not args[:-1]:
                args_list, alens, axmm = [], [], []
            else:
                args_list, alens, axmm = list(
                    zip(*[cls._args(v.strip()) for v in args[:-1].split(",")])
                )
                
            cregs, goregs, idxs = [], [], [0, 0]
            
            for xmm in axmm:
                key = 0 if xmm else 1
                seq = FPARGS_ORDER if xmm else ARGS_ORDER_C
                goseq = FPARGS_ORDER if xmm else ARGS_ORDER_GO
                
                if idxs[key] >= len(seq):
                    raise cls._err("too many arguments")
                    
                cregs.append(seq[idxs[key]])
                goregs.append(goseq[idxs[key]])
                idxs[key] += 1
                
            proto = Prototype(
                retv,
                [
                    Parameter(arg, size, creg, goreg)
                    for arg, size, creg, goreg in zip(args_list, alens, cregs, goregs)
                ],
            )
            ret[name] = proto
            log.debug(f"Parsed prototype: func {name}{repr(proto)}")
            
        return pkg, ret


class Pcsp:
    """程序计数器到栈指针映射"""

    entry: int
    maxpc: int
    out: List[Tuple[int, int]]
    pc: int
    sp: int

    def __init__(self, entry: int):
        self.out, self.maxpc, self.entry, self.pc, self.sp = [], entry, entry, entry, 0

    def __str__(self) -> str:
        ret = "[][2]uint32{\n"
        for pc, sp in self.out:
            ret += f"        {{{pc}, {sp}}},\n"
        return ret + "    }"

    def optimize(self):
        self.out.append((self.pc - self.entry, self.sp))
        self.out.sort(key=lambda x: x[0])
        tmp, lpc, lsp = [(1, 0)], 0, -1
        for pc, sp in self.out:
            if pc != lpc and sp != lsp:
                tmp.append((pc, sp))
            if pc != lpc and sp == lsp:
                if len(tmp) > 0:
                    tmp.pop(-1)
                tmp.append((pc, sp))
            lpc, lsp = pc, sp
        self.out = tmp

    def update(self, dpc: int, dsp: int):
        self.out.append((self.pc - self.entry, self.sp))
        self.pc += dpc
        self.sp += dsp
        if self.pc > self.maxpc:
            self.maxpc = self.pc


class Instruction:
    """RISC-V指令"""

    asm_code: str
    mnemonic: str
    data: Union[bytes, str]
    fixups: List
    offs_: Optional[int]
    label_name: Optional[str]

    def __init__(self, line: str):
        self.asm_code = line.strip()
        self.mnemonic = self.asm_code.split(None, 1)[0] if self.asm_code else ""
        self.data, self.fixups, self.offs_, self.label_name = b"", [], None, None

    @property
    def size(self) -> int:
        return 4  # RISC-V instructions are always 4 bytes

    @property
    def is_branch(self) -> bool:
        return self.mnemonic.startswith("b") or self.mnemonic in [
            "j", "jal", "jalr", "ret", "call", "tail"
        ]

    @property
    def is_return(self) -> bool:
        return self.mnemonic == "ret"

    @property
    def is_jmp(self) -> bool:
        return self.mnemonic == "j"

    @property
    def is_invoke(self) -> bool:
        return self.mnemonic in ["call", "jal", "jalr"]

    @property
    def is_branch_label(self) -> bool:
        return self.is_branch and self.label_name is not None

    @property
    def need_reloc(self) -> bool:
        return len(self.fixups) != 0

    def set_label_offset(self, off):
        self.offs_ = off

    @functools.cached_property
    def encoded(self) -> str:
        if isinstance(self.data, str):
            return self.data
        if self.data:
            (val,) = struct.unpack("<I", self.data[:4])
            return f"WORD $0x{val:08x}"
        return f"// SKIPPED: {self.asm_code}"

    @staticmethod
    def encode(buf: bytes, comments: str = "") -> str:
        if not buf:
            return f"// {comments}" if comments else ""
        if len(buf) % 4 != 0:
            raise RuntimeError(f"Instruction not 4-byte aligned: {comments}")
        (val,) = struct.unpack("<I", buf[:4])
        r = f"WORD $0x{val:08x}"
        return f"{r}  // {comments}" if comments else r


class Instr:
    """指令基类"""

    len: int = NotImplemented
    instr: Union[str, Instruction] = NotImplemented

    def size(self, _: int) -> int:
        return self.len

    def formatted(self, pc: int) -> str:
        raise NotImplementedError

    @staticmethod
    def raw_formatted(bs: bytes, comm: str, pc: Optional[int]) -> str:
        t = "\t"
        if bs:
            t += ", ".join([f"0x{b:02x}" for b in bs]) + ", "
        return f"{t}//{(f'0x{pc:08x} ' if pc is not None else ' ')}{comm}"


class RawInstr(Instr):
    """原始字节指令"""

    bs: bytes

    def __init__(self, size: int, instr: str, bs: bytes):
        self.len, self.instr, self.bs = size, instr, bs

    def formatted(self, _: int) -> str:
        return "\t" + self.instr

    def raw_formatted(self, pc: int) -> str:
        return Instr.raw_formatted(self.bs, self.instr, pc)


class IntInstr(Instr):
    """整数指令"""

    comm: str
    func: Callable[[], int]

    def __init__(self, size: int, func: Callable[[], int], comments: str = ""):
        self.len, self.func, self.comm = size, func, comments

    @property
    def raw_bytes(self):
        return self.func().to_bytes(self.len, "little")

    @property
    def instr(self) -> str:
        return Instruction.encode(self.raw_bytes, self.comm)

    def formatted(self, _: int) -> str:
        return "\t" + self.instr


class RVInstr(Instr):
    """RISC-V指令"""

    def __init__(self, instr: Instruction):
        self.len, self.instr = instr.size, instr

    def formatted(self, _: int) -> str:
        encoded = self.instr.encoded
        if encoded.startswith("WORD"):
            return f"\t{encoded}  // {self.instr.asm_code}"
        return f"\t{encoded}"

    def raw_formatted(self, pc: int) -> str:
        return Instr.raw_formatted(
            self.instr.data if isinstance(self.instr.data, bytes) else b"",
            str(self.instr),
            pc,
        )


class LabelInstr(Instr):
    """标签指令"""

    def __init__(self, name: str):
        self.len, self.instr = 0, name

    def formatted(self, _: int) -> str:
        if self.instr.isidentifier():
            return self.instr + ":"
        return f"_LB_{hash(self.instr) & 0xFFFFFFFF:08x}: // {self.instr}"


class CommentInstr(Instr):
    """注释指令"""

    def __init__(self, text: str):
        self.len, self.instr = 0, "// " + text

    def formatted(self, _: int) -> str:
        return "\t" + self.instr


class AlignmentInstr(Instr):
    """对齐指令"""

    bits: int
    fill: int

    def __init__(self, bits: int, fill: int = 0):
        self.bits, self.fill = bits, fill

    def size(self, pc: int) -> int:
        mask = (1 << self.bits) - 1
        return (mask - (pc & mask) + 1) & mask

    def formatted(self, pc: int) -> str:
        buf = bytes([self.fill]) * self.size(pc)
        return "\t" + Instruction.encode(
            buf, f".p2align {self.bits}, 0x{self.fill:02x}"
        )


class Counter:
    """计数器"""

    value: int = 0

    @classmethod
    def next(cls) -> int:
        val, cls.value = cls.value, cls.value + 1
        return val


class BasicBlock:
    """基本块"""

    maxsp: int
    name: str
    weak: bool
    func: bool
    body: List[Instr]
    prevs: List["BasicBlock"]
    next: Optional["BasicBlock"]
    jump: Optional["BasicBlock"]

    def __init__(self, name: str, weak: bool = True, func: bool = False):
        (
            self.maxsp,
            self.body,
            self.prevs,
            self.name,
            self.weak,
            self.next,
            self.jump,
            self.func,
        ) = (-1, [], [], name, weak, None, None, func)

    def __repr__(self):
        return f"{{BasicBlock {repr(self.name)}}}"

    @property
    def last(self) -> Optional[Instr]:
        return next(
            (v for v in reversed(self.body) if not isinstance(v, CommentInstr)), None
        )

    def size_of(self, pc: int) -> int:
        return functools.reduce(lambda p, v: p + v.size(pc + p), self.body, 0)

    def link_to(self, block: "BasicBlock"):
        self.next = block
        block.prevs.append(self)

    def jump_to(self, block: "BasicBlock"):
        self.jump = block
        block.prevs.append(self)

    @classmethod
    def anonymous(cls) -> "BasicBlock":
        return cls(f"// bb.{Counter.next()}", weak=False)


class CodeSection:
    """代码段"""

    dead: bool
    export: bool
    blocks: List[BasicBlock]
    labels: Dict[str, BasicBlock]
    funcs: Dict[str, Pcsp]
    bsmap_: Dict[str, int]

    def __init__(self):
        self.dead, self.labels, self.export, self.blocks, self.funcs, self.bsmap_ = (
            False,
            {},
            False,
            [BasicBlock.anonymous()],
            {},
            {},
        )

    @property
    def block(self) -> BasicBlock:
        return self.blocks[-1]

    @property
    def instrs(self) -> Iterable[Instr]:
        for block in self.blocks:
            yield from block.body

    def _make(self, name: str, func: bool = False):
        if func and (old := self.labels.get(name)) and (old.func != func):
            old.func = True
        return self.labels.setdefault(name, BasicBlock(name, func=func))

    def _next(self, link: BasicBlock):
        if self.dead:
            self.dead = False
        else:
            self.block.link_to(link)

    def _decl(self, name: str, block: BasicBlock):
        block.weak = False
        block.body.append(LabelInstr(name))
        self._next(block)
        self.blocks.append(block)

    def _kill(self, name: str):
        self.dead = True
        self.block.link_to(self._make(name))

    def _split(self, jmp: BasicBlock):
        link = BasicBlock.anonymous()
        self.labels[link.name] = link
        self.block.link_to(link)
        self.block.jump_to(jmp)
        self.blocks.append(link)

    @staticmethod
    def _mk_align(v: int) -> int:
        if v & 15 == 0:
            return v
        log.warning("SP is not aligned to 16 bytes.")
        return (v + 15) & -16

    def _find_label(self, name: str, adjs: Iterable[int], size: int = 0) -> int:
        for adj, block in zip(adjs, self.blocks):
            if block.name == name:
                return size
            v = self.bsmap_.get(block.name)
            if v is not None:
                size += v + adj
            else:
                block_size = block.size_of(size)
                size += block_size + adj
                self.bsmap_[block.name] = block_size
        raise SyntaxError("unresolved reference to name: " + name)

    def _check_split(self, instr: Instruction):
        if instr.is_return:
            log.debug(f"Return instruction '{instr.asm_code}' marks end of block.")
            self.dead = True
        elif instr.is_branch and instr.label_name:
            log.debug(
                f"Branch instruction '{instr.asm_code}' to '{instr.label_name}' creates control flow split."
            )
            if instr.is_jmp:
                self._kill(instr.label_name)
            elif instr.is_invoke:
                self._split(self._make(instr.label_name, func=True))
            else:
                self._split(self._make(instr.label_name))

    def _trace_block(self, bb: BasicBlock, pcsp: Optional[Pcsp]) -> int:
        if pcsp is not None:
            if bb.name in self.funcs:
                pcsp = None
            else:
                pcsp.pc = self.get(bb.name)
                if bb.func or pcsp.pc < pcsp.entry:
                    pcsp = Pcsp(pcsp.pc)
                    self.funcs[bb.name] = pcsp
        if bb.maxsp == -1:
            return self._trace_nocache(bb, pcsp)
        if bb.maxsp >= 0:
            return bb.maxsp
        return 0

    def _trace_nocache(self, bb: BasicBlock, pcsp: Optional[Pcsp]) -> int:
        log.debug(f"Tracing stack for block '{bb.name}'...")
        bb.maxsp = -2
        if pcsp:
            pc0, sp0 = pcsp.pc, pcsp.sp
        maxsp, term = self._trace_instructions(bb, pcsp)
        if term:
            log.debug(f"Block '{bb.name}' is terminal. Max SP in block: {maxsp}")
            return maxsp
        a, b = 0, 0
        if pcsp:
            pc, sp = pcsp.pc, pcsp.sp
        if bb.jump:
            log.debug(f"  ...recursing to jump target '{bb.jump.name}'")
            a = self._trace_block(bb.jump, pcsp)
            if pcsp:
                pcsp.pc, pcsp.sp = pc, sp
        if bb.next:
            log.debug(f"  ...recursing to next block '{bb.next.name}'")
            b = self._trace_block(bb.next, pcsp)
            if pcsp:
                pcsp.pc, pcsp.sp = pc, sp
        if pcsp:
            pcsp.pc, pcsp.sp = pc0, sp0
        bb.maxsp = maxsp + max(a, b)
        log.debug(f"Finished tracing block '{bb.name}'. Cumulative max SP: {bb.maxsp}")
        return bb.maxsp

    def _trace_instructions(self, bb: BasicBlock, pcsp: Pcsp) -> Tuple[int, bool]:
        cursp, maxsp, close = 0, 0, False
        for ins in bb.body:
            diff = 0
            if isinstance(ins, RVInstr):
                instr_obj = ins.instr
                if instr_obj.is_return:
                    close = True
                # Track stack pointer changes for RISC-V
                if instr_obj.mnemonic == "addi":
                    parts = instr_obj.asm_code.replace(",", " ").split()
                    if len(parts) >= 4 and parts[1] == "sp" and parts[2] == "sp":
                        try:
                            diff = -self._mk_align(int(parts[3]))
                        except ValueError:
                            pass
                cursp += diff
                if cursp > maxsp:
                    maxsp = cursp
            if pcsp:
                pcsp.update(ins.size(pcsp.pc), diff)
        return maxsp, close

    def get(self, key: str) -> Optional[int]:
        if key not in self.labels:
            raise SyntaxError(f"unresolved reference to name: {key}")
        return self._find_label(key, itertools.repeat(0, len(self.blocks)))

    def has(self, key: str) -> bool:
        return key in self.labels

    def emit(self, buf: bytes, comments: str = ""):
        if not self.dead:
            self.block.body.append(
                RawInstr(len(buf), Instruction.encode(buf, comments or buf.hex()), buf)
            )

    def lazy(self, size: int, func: Callable[[], int], comments: str = ""):
        if not self.dead:
            self.block.body.append(IntInstr(size, func, comments))

    def label(self, name: str):
        log.debug(f"Defining label: '{name}'")
        if name not in self.labels or self.labels[name].weak:
            self._decl(name, self._make(name))
        else:
            raise SyntaxError("duplicated label: " + name)

    def instr(self, instr: Instruction):
        if not self.dead:
            self.block.body.append(RVInstr(instr))
            self._check_split(instr)

    def stacksize(self, name: str) -> int:
        if name not in self.labels:
            raise SyntaxError("undefined function: " + name)
        log.info(f"Calculating stack size for function '{name}'...")
        size = self._trace_block(self.labels[name], None)
        log.info(f"Calculated stack size for '{name}' is {size} bytes.")
        return size

    def pcsp(self, name: str, entry: int) -> int:
        if name not in self.labels:
            raise SyntaxError("undefined function: " + name)
        pcsp = Pcsp(entry)
        self.labels[name].func = True
        return self._trace_block(self.labels[name], pcsp)

# ============================================================================
# RISC-V 汇编器主类
# ============================================================================

REG_MAP = {
    "a0": ("MOV", "X10"),
    "a1": ("MOV", "X11"),
    "a2": ("MOV", "X12"),
    "a3": ("MOV", "X13"),
    "a4": ("MOV", "X14"),
    "a5": ("MOV", "X15"),
    "a6": ("MOV", "X16"),
    "a7": ("MOV", "X17"),
    "fa0": ("FMOVD", "F0"),
    "fa1": ("FMOVD", "F1"),
    "fa2": ("FMOVD", "F2"),
    "fa3": ("FMOVD", "F3"),
    "fa4": ("FMOVD", "F4"),
    "fa5": ("FMOVD", "F5"),
    "fa6": ("FMOVD", "F6"),
    "fa7": ("FMOVD", "F7"),
}

GNU_TO_GO_REG = {
    "zero": "ZERO",
    "ra": "X1", 
    "sp": "X2",
    "gp": "X3",
    "tp": "X4",
    "t0": "X5",
    "t1": "X6",
    "t2": "X7",
    "s0": "X8",
    "fp": "X8",
    "s1": "X9",
    "a0": "X10",
    "a1": "X11",
    "a2": "X12",
    "a3": "X13",
    "a4": "X14",
    "a5": "X15",
    "a6": "X16",
    "a7": "X17",
    "s2": "X18",
    "s3": "X19",
    "s4": "X20",
    "s5": "X21",
    "s6": "X22",
    "s7": "X23",
    "s8": "X24",
    "s9": "X25",
    "s10": "X26",
    "s11": "g",
    "t3": "X28",
    "t4": "X29",
    "t5": "X30",
    "t6": "X31",
}

# Add numeric register names
for i in range(32):
    GNU_TO_GO_REG[f"x{i}"] = GNU_TO_GO_REG.get(
        {
            0: "zero", 1: "ra", 2: "sp", 3: "gp", 4: "tp",
            5: "t0", 6: "t1", 7: "t2", 8: "s0", 9: "s1",
            10: "a0", 11: "a1", 12: "a2", 13: "a3", 14: "a4",
            15: "a5", 16: "a6", 17: "a7", 18: "s2", 19: "s3",
            20: "s4", 21: "s5", 22: "s6", 23: "s7", 24: "s8",
            25: "s9", 26: "s10", 27: "s11", 28: "t3", 29: "t4",
            30: "t5", 31: "t6",
        }.get(i, f"X{i}")
    )

STUB_NAME = "native_entry"
STUB_SIZE = 67
WITH_OFFS = os.getenv("ASM2ASM_DEBUG_OFFSET", "").lower() in ("1", "yes", "true")
OUTPUT_RAW = False


class Assembler:
    """RISC-V汇编器"""

    out: List[str]
    subr: Dict[str, int]
    code: CodeSection
    vals: Dict[str, Union[str, int]]
    pending_lui: Dict[str, str]
    entry_point_name: Optional[str]

    def __init__(self):
        self.out, self.subr, self.vals, self.code, self.pending_lui = (
            [],
            {},
            {},
            CodeSection(),
            {},
        )
        self.entry_point_name = None
        log.info("Assembler initialized.")

    def _get(self, v: str) -> int:
        if v not in self.vals:
            return self.code.get(v)
        if isinstance(self.vals[v], int):
            return self.vals[v]
        ret = self.vals[v] = self._eval(self.vals[v])
        return ret

    def _eval(self, v: str) -> int:
        return Expression(v).eval(self._get)

    def _emit(self, v: bytes, cmd: str):
        align_size = len(v) % 4
        if align_size != 0:
            v += int(0).to_bytes(4 - align_size, "little")
        for i in range(0, len(v), 4):
            self.code.emit(
                v[i : i + 4], f"{cmd} {len(v[i:i+4])}, {repr(v[i:i+16])[1:]}"
            )

    def _limit(self, v: int, a: int, b: int) -> int:
        if not (a <= v <= b):
            raise SyntaxError(f"integer constant out of bound [{a}, {b}): {v}")
        return v

    def _assemble_instruction(self, ins: str) -> bytes:
        """Simplified llvm-mc wrapper for RISC-V instruction assembly"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.s', delete=False) as asm_file:
                asm_file.write(ins)
                asm_file.flush()
                
                # Simple llvm-mc call
                result = subprocess.run([
                    'llvm-mc', 
                    '-arch=riscv64',
                    '-mattr=+m,+a,+f,+d,+v',  # Basic extensions
                    '--show-encoding',
                    asm_file.name
                ], capture_output=True, text=True, timeout=3)
                
                if result.returncode != 0:
                    raise RuntimeError(f"llvm-mc failed: {result.stderr}")
                
                # Extract hex bytes from output like "# encoding: [0x13,0x00,0x00,0x00]"
                hex_pattern = r'# encoding: \[(0x[0-9a-f,x\s]+)\]'
                match = re.search(hex_pattern, result.stdout)
                if match:
                    hex_str = match.group(1)
                    # Parse hex bytes: "0x13,0x00,0x00,0x00" -> bytes
                    hex_bytes = [int(x.strip(), 16) for x in hex_str.split(',') if x.strip()]
                    return bytes(hex_bytes)
                
                raise ValueError("Could not extract encoding from llvm-mc output")
                
        except Exception as e:
            log.debug(f"Binary assembly failed for '{ins}': {e}")
            return b'\x13\x00\x00\x00'  # Default to NOP instruction
        finally:
            if 'asm_file' in locals() and os.path.exists(asm_file.name):
                os.unlink(asm_file.name)

    def _translate_reg(self, reg: str) -> str:
        """Translate GNU RISC-V register name to Go assembly register name"""
        reg = reg.strip().lower()
        
        # Remove % prefix if present
        if reg.startswith('%'):
            reg = reg[1:]
        
        # Handle floating point registers
        if reg.startswith('f'):
            if reg in ['fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5', 'fa6', 'fa7']:
                return f"F{reg[2:]}"
            elif reg.startswith('ft'):
                return f"F{5 + int(reg[2:])}" if reg[2:].isdigit() else reg.upper()
            elif reg.startswith('fs'):
                return f"F{8 + int(reg[2:])}" if reg[2:].isdigit() else reg.upper()
            else:
                return reg.upper()
        
        # Use the mapping table
        return GNU_TO_GO_REG.get(reg, reg.upper())

    def _get_go_load_mnemonic(self, riscv_mnemonic: str) -> str:
        """Convert RISC-V load mnemonic to Go assembly equivalent"""
        mapping = {
            'ld': 'MOV',      # 64-bit load
            'lw': 'MOVW',     # 32-bit load (sign-extended)
            'lwu': 'MOVWU',   # 32-bit load (zero-extended)  
            'lh': 'MOVH',     # 16-bit load (sign-extended)
            'lhu': 'MOVHU',   # 16-bit load (zero-extended)
            'lb': 'MOVB',     # 8-bit load (sign-extended)
            'lbu': 'MOVBU',   # 8-bit load (zero-extended)
            'fld': 'FMOVD',   # 64-bit float load
            'flw': 'FMOVS',   # 32-bit float load
        }
        return mapping.get(riscv_mnemonic, 'MOV')

    def _get_go_store_mnemonic(self, riscv_mnemonic: str) -> str:
        """Convert RISC-V store mnemonic to Go assembly equivalent"""
        mapping = {
            'sd': 'MOV',      # 64-bit store
            'sw': 'MOVW',     # 32-bit store
            'sh': 'MOVH',     # 16-bit store
            'sb': 'MOVB',     # 8-bit store
            'fsd': 'FMOVD',   # 64-bit float store
            'fsw': 'FMOVS',   # 32-bit float store
        }
        return mapping.get(riscv_mnemonic, 'MOV')

    def _get_go_branch_mnemonic(self, riscv_mnemonic: str) -> str:
        """Convert RISC-V branch mnemonic to Go assembly equivalent"""
        mapping = {
            'beq': 'BEQ',
            'bne': 'BNE', 
            'blt': 'BLT',
            'bge': 'BGE',
            'bltu': 'BLTU',
            'bgeu': 'BGEU',
            'j': 'JMP',
            'jal': 'CALL',
            'jalr': 'CALL',
            'b': 'JMP',
        }
        return mapping.get(riscv_mnemonic, riscv_mnemonic.upper())

    def _get_go_arithmetic_op(self, riscv_op: str) -> str:
        """Convert RISC-V arithmetic operation to Go assembly equivalent"""
        mapping = {
            'add': 'ADD',
            'addi': 'ADD',
            'sub': 'SUB',
            'and': 'AND',
            'andi': 'AND',
            'or': 'OR',
            'ori': 'OR',
            'xor': 'XOR',
            'xori': 'XOR',
            'sll': 'SLL',
            'slli': 'SLL',
            'srl': 'SRL',
            'srli': 'SRL',
            'sra': 'SRA',
            'srai': 'SRA',
        }
        return mapping.get(riscv_op, riscv_op.upper())

    def _translate_operand(self, operand: str) -> str:
        """Translate a RISC-V operand to Go assembly format"""
        operand = operand.strip()
        
        # Immediate values
        if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
            return f"${operand}"
        
        # Hexadecimal immediate
        if operand.startswith('0x'):
            return f"${operand}"
        
        # Register
        if operand in GNU_TO_GO_REG:
            return self._translate_reg(operand)
        
        # Default case
        return operand

    def _parse_immediate(self, imm_str: str) -> str:
        """Parse immediate value from string"""
        imm_str = imm_str.strip()
        if imm_str.isdigit() or (imm_str.startswith('-') and imm_str[1:].isdigit()):
            return imm_str
        if imm_str.startswith('0x'):
            return str(int(imm_str, 16))
        return "0"  # Default to 0 for unparseable immediates

    def _extract_branch_target(self, operands_str: str) -> Optional[str]:
        """Extract branch target label from operands string"""
        # For conditional branches like "beq x1, x2, label"
        parts = [p.strip() for p in operands_str.split(',')]
        target = parts[-1]  # Last operand is usually the target
        
        # Check if it looks like a label (starts with . or is identifier)
        if target.startswith('.') or target.replace('_', '').replace('.', '').isalnum():
            return target.lstrip('.')
        
        return None

    def _get_branch_args(self, operands_str: str, mnemonic: str) -> str:
        """Get the arguments for branch instruction (registers before the label)"""
        parts = [p.strip() for p in operands_str.split(',')]
        
        if mnemonic in ['j', 'jal'] or mnemonic.startswith('b'):
            # Remove the last part (label) and translate remaining registers
            reg_parts = parts[:-1] if len(parts) > 1 else []
            translated_regs = [self._translate_reg(reg) for reg in reg_parts]
            return ', '.join(translated_regs) + (', ' if translated_regs else '')
        
        return ''

    def _try_pattern_match(self, clean_line: str, original_line: str) -> Optional[str]:
        """Try to match common RISC-V patterns and convert to Go assembly"""
        
        # 1. LUI/ADDI symbol loading pairs
        lui_match = re.match(r"lui\s+([a-z0-9]+),\s*%hi\(([_a-zA-Z0-9\.]+)\)", clean_line)
        if lui_match:
            reg, symbol = lui_match.groups()
            self.pending_lui[reg] = symbol
            return f"MOV $·{symbol.lstrip('.')}(SB), {self._translate_reg(reg)}"
        
        addi_match = re.match(r"addi\s+([a-z0-9]+),\s*\1,\s*%lo\(([_a-zA-Z0-9\.]+)\)", clean_line)
        if addi_match:
            reg, symbol = addi_match.groups()
            if self.pending_lui.get(reg) == symbol:
                del self.pending_lui[reg]
                return ""  # Skip, already handled in LUI
        
        # 2. Function calls
        call_match = re.match(r"(call|tail)\s+([_a-zA-Z0-9\.]+)", clean_line)
        if call_match:
            mnemonic, symbol = call_match.groups()
            go_mnemonic = "CALL" if mnemonic == "call" else "JMP"
            return f"{go_mnemonic} ·{symbol.lstrip('.')}(SB)"
        
        # 3. Simple register moves
        if clean_line.startswith("mv "):
            parts = clean_line[3:].split(',')
            if len(parts) == 2:
                src, dst = parts[0].strip(), parts[1].strip()
                return f"MOV {self._translate_reg(src)}, {self._translate_reg(dst)}"
        
        # 4. Return
        if clean_line == "ret":
            return "RET"
        
        # 5. NOP
        if clean_line == "nop":
            return "NOP"
        
        # 6. Memory loads
        load_match = re.match(r"(ld|lw|lh|lb|lbu|lhu|lwu|fld|flw)\s+([a-z0-9]+),\s*([^(]*)\(([a-z0-9]+)\)", clean_line)
        if load_match:
            mnemonic, dst_reg, offset, base_reg = load_match.groups()
            
            # Handle symbol references like %lo(symbol)
            if "%lo(" in offset:
                symbol_match = re.search(r"%lo\(([^)]+)\)", offset)
                if symbol_match:
                    symbol = symbol_match.group(1)
                    if self.pending_lui.get(base_reg) == symbol:
                        del self.pending_lui[base_reg]
                        go_mnemonic = self._get_go_load_mnemonic(mnemonic)
                        return f"{go_mnemonic} ({self._translate_reg(base_reg)}), {self._translate_reg(dst_reg)}"
            
            # Regular memory load
            go_mnemonic = self._get_go_load_mnemonic(mnemonic)
            offset_val = self._parse_immediate(offset) if offset.strip() else "0"
            return f"{go_mnemonic} {offset_val}({self._translate_reg(base_reg)}), {self._translate_reg(dst_reg)}"
        
        # 7. Memory stores
        store_match = re.match(r"(sd|sw|sh|sb|fsd|fsw)\s+([a-z0-9]+),\s*([^(]*)\(([a-z0-9]+)\)", clean_line)
        if store_match:
            mnemonic, src_reg, offset, base_reg = store_match.groups()
            go_mnemonic = self._get_go_store_mnemonic(mnemonic)
            offset_val = self._parse_immediate(offset) if offset.strip() else "0"
            return f"{go_mnemonic} {self._translate_reg(src_reg)}, {offset_val}({self._translate_reg(base_reg)})"
        
        # 8. Arithmetic operations
        arith_match = re.match(r"(add|sub|and|or|xor|sll|srl|sra)i?\s+([^,]+),\s*([^,]+)(?:,\s*(.+))?", clean_line)
        if arith_match:
            op, dst, src1, src2 = arith_match.groups()
            go_op = self._get_go_arithmetic_op(op)
            if src2 and src2.strip():  # Three operand form
                return f"{go_op} {self._translate_operand(src1)}, {self._translate_operand(src2)}, {self._translate_reg(dst)}"
            else:  # Two operand form (immediate)
                return f"{go_op} {self._translate_operand(src1)}, {self._translate_reg(dst)}"
        
        return None  # No pattern matched

    def _process_instruction(self, line: str) -> Instruction:
        """Process RISC-V instruction with hybrid approach: pattern matching + binary fallback"""
        instr = Instruction(line)
        clean_line = line.strip().lower()
        
        # Extract label if this is a branch instruction
        if instr.is_branch:
            instr.label_name = self._extract_branch_target(clean_line.split(None, 1)[1] if ' ' in clean_line else '')
        
        # Try pattern matching first (fast path)
        if translated := self._try_pattern_match(clean_line, line):
            instr.data = translated
            return instr
        
        # Fallback to binary assembly (slow path) for complex instructions
        try:
            binary_data = self._assemble_instruction(line)
            instr.data = Instruction.encode(binary_data, line)
        except Exception as e:
            log.warning(f"Failed to process '{line}': {e}")
            instr.data = f"WORD $0x00000013  // {line} (failed)"
        
        return instr

    def _cmd_nop(self, _: List[str]):
        pass

    def _cmd_set(self, args: List[str]):
        if len(args) != 2:
            raise SyntaxError(".set takes 2 arguments")
        if not args[0].isidentifier():
            raise SyntaxError(f"{repr(args[0])} is not a valid identifier")
        self.vals[args[0]] = args[1]
        log.debug(f"Handled .set {args[0]} = {args[1]}")

    def _cmd_byte(self, args: List[str]):
        if len(args) != 1:
            raise SyntaxError(".byte takes 1 argument")
        self.code.lazy(
            1,
            lambda: self._limit(self._eval(args[0]), -0x80, 0xFF) & 0xFF,
            f".byte {args[0]}",
        )

    def _cmd_word(self, args: List[str]):
        if len(args) != 1:
            raise SyntaxError(".word takes 1 argument")
        self.code.lazy(
            2,
            lambda: self._limit(self._eval(args[0]), -0x8000, 0xFFFF) & 0xFFFF,
            f".word {args[0]}",
        )

    def _cmd_long(self, args: List[str]):
        if len(args) != 1:
            raise SyntaxError(".long takes 1 argument")
        self.code.lazy(
            4,
            lambda: self._limit(self._eval(args[0]), -0x80000000, 0xFFFFFFFF)
            & 0xFFFFFFFF,
            f".long {args[0]}",
        )

    def _cmd_quad(self, args: List[str]):
        if len(args) != 1:
            raise SyntaxError(".quad takes 1 argument")
        self.code.lazy(
            8,
            lambda: self._limit(
                self._eval(args[0]), -0x8000000000000000, 0xFFFFFFFFFFFFFFFF
            )
            & 0xFFFFFFFFFFFFFFFF,
            f".quad {args[0]}",
        )

    def _cmd_ascii(self, args: List[str]):
        if len(args) != 1:
            raise SyntaxError(".ascii takes 1 argument")
        self._emit(args[0].encode("latin-1"), ".ascii")

    def _cmd_asciz(self, args: List[str]):
        if len(args) != 1:
            raise SyntaxError(".asciz takes 1 argument")
        self._emit(args[0].encode("latin-1") + b"\0", ".asciz")

    def _cmd_space(self, args: List[str]):
        nb = self._eval(args[0])
        fv = self._limit(self._eval(args[1]), 0, 255) if len(args) > 1 else 0
        self._emit(bytes([fv] * nb), ".space")

    def _cmd_p2align(self, args: List[str]):
        bits = self._eval(args[0])
        fill = self._eval(args[1]) if len(args) > 1 else 0
        self.code.block.body.append(AlignmentInstr(bits, fill))
        log.debug(f"Handled .p2align {bits}")

    @functools.cached_property
    def _commands(self) -> dict:
        return {
            ".set": self._cmd_set,
            ".int": self._cmd_long,
            ".long": self._cmd_long,
            ".byte": self._cmd_byte,
            ".quad": self._cmd_quad,
            ".word": self._cmd_word,
            ".hword": self._cmd_word,
            ".short": self._cmd_word,
            ".ascii": self._cmd_ascii,
            ".asciz": self._cmd_asciz,
            ".space": self._cmd_space,
            ".globl": self._cmd_nop,
            ".text": self._cmd_nop,
            ".file": self._cmd_nop,
            ".type": self._cmd_nop,
            ".p2align": self._cmd_p2align,
            ".align": self._cmd_nop,
            ".size": self._cmd_nop,
            ".section": self._cmd_nop,
            ".attribute": self._cmd_nop,
        }

    @staticmethod
    def _remove_comments(line: str) -> str:
        return line.split("//")[0].split("#")[0]

    def _parse(self, src: List[str]):
        log.info("Starting assembly parsing phase...")
        is_text_section = False
        
        for i, line in enumerate(src):
            line = self._remove_comments(line).strip()
            if not line:
                continue
                
            log.debug(f"Parsing line {i+1}: '{line}'")
            
            if line == ".text":
                is_text_section = True
                log.debug("Entered .text section.")
                continue
                
            if line.startswith(".globl"):
                potential_entry = line.split(None, 1)[1].strip()
                if is_text_section and self.entry_point_name is None:
                    self.entry_point_name = potential_entry
                    log.info(f"Identified potential entry point from .globl: '{self.entry_point_name}'")
                continue
                
            if line.endswith(":"):
                label_name = line[:-1]
                self.code.label(label_name)
                if is_text_section and self.entry_point_name == label_name:
                    log.info(f"Confirmed entry point label: '{self.entry_point_name}'")
                continue
                
            if line.startswith("."):
                cmd = Command.parse(line)
                if func := self._commands.get(cmd.cmd):
                    func(cmd.args)
                else:
                    log.warning(f"Ignoring unknown directive: {cmd.cmd}")
                continue
                
            # Process regular instructions
            instr = self._process_instruction(line)
            if instr.data != "":
                self.code.instr(instr)
                
        log.info("Assembly parsing finished.")

    def _reloc(self, rip: int = 0):
        log.info("Performing relocation pass (calculating PC for each instruction)...")
        for block in self.code.blocks:
            for instr in block.body:
                rip += instr.size(rip)
        log.info("Relocation pass finished.")

    def _declare(self, protos: PrototypeMap):
        log.info("Starting declaration and code generation phase...")
        
        if self.entry_point_name is None:
            raise RuntimeError("Could not determine the entry point function from the assembly file.")
            
        proto_name = None
        if self.entry_point_name in protos:
            proto_name = self.entry_point_name
        elif f"_{self.entry_point_name}" in protos:
            proto_name = f"_{self.entry_point_name}"
        elif f"__{self.entry_point_name}" in protos:
            proto_name = f"__{self.entry_point_name}"
            
        if proto_name is None:
            raise RuntimeError(
                f"Assembly entry point '{self.entry_point_name}' not found in prototype file. "
                f"Searched for '{self.entry_point_name}', '_{self.entry_point_name}', and '__{self.entry_point_name}'."
            )
            
        log.info(f"Matched assembly entry point '{self.entry_point_name}' with prototype '{proto_name}'.")
        self._declare_body(self.entry_point_name)
        self._declare_functions(protos)
        log.info("Declaration and code generation finished.")

    def _declare_body(self, asm_name: str):
        size = self.code.stacksize(asm_name)
        self.out.append(f"TEXT ·_{asm_name}_entry__(SB), NOSPLIT, ${size}")
        self.out.append("\tNO_LOCAL_POINTERS")
        self._reloc()
        
        pc = 0
        for v in self.code.instrs:
            self.out.append((f"// +{pc}\n" if WITH_OFFS else "") + v.formatted(pc))
            pc += v.size(pc)

    def _declare_function(self, name: str, proto: Prototype):
        asm_name = name.lstrip("_")
        log.info(f"Generating wrapper for function '·{name}(SB)'.")
        
        if not self.code.has(asm_name):
            log.warning(
                f"Function '{asm_name}' (from prototype '{name}') not found in assembly code, skipping wrapper generation."
            )
            return
            
        addr = self.code.get(asm_name)
        self.subr[asm_name] = addr
        size = self.code.stacksize(asm_name)
        
        log.debug(f"  - Stack size: {size}, Arg space: {proto.argspace}")
        
        self.out.append("")
        self.out.append(f"TEXT ·{name}(SB), NOSPLIT, ${size}-{proto.argspace}")
        self.out.append("\tNO_LOCAL_POINTERS")
        
        # Add stack check if needed
        if size > 0:
            self.out.append(f"_{name}_entry:")
            self.out.append("\tMOV g, X27")
            self.out.append("\tMOV 16(X27), X5")
            
            if size < 2048:
                self.out.append(f"\tADD $-{size}, X2, X6")  # X2 is SP
            else:
                self.out.append(f"\tMOV $-{size}, X6")
                self.out.append("\tADD X2, X6, X6")
                
            self.out.append(f"\tBLTU X6, X5, _{name}_stack_grow")
        
        # Initialize all the arguments
        offs = 0
        for arg in proto.args:
            op, reg = REG_MAP[arg.creg.reg]
            self.out.append(f"\t{op} {arg.name}+{offs}(FP), {reg}")
            offs += arg.size
        
        # Call the actual function
        self.out.append(f"\tCALL ·_{asm_name}_entry__(SB)")
        
        # Handle return value
        if proto.retv is not None:
            op, reg = REG_MAP[proto.retv.creg.reg]
            self.out.append(f"\t{op} {reg}, {proto.retv.name}+{offs}(FP)")
        
        self.out.append("\tRET")
        
        # Add stack growing code if needed
        if size > 0:
            self.out.append(f"_{name}_stack_grow:")
            self.out.append("\tMOV X1, X3")  # X1 is RA (return address)
            self.out.append("\tCALL runtime·morestack_noctxt<>(SB)")
            self.out.append(f"\tJMP _{name}_entry")

    def _declare_functions(self, protos: PrototypeMap):
        for name, proto in sorted(protos.items()):
            if name.startswith("_"):
                self._declare_function(name, proto)
            else:
                raise SyntaxError('function prototype must have a "_" prefix: ' + repr(name))

    def parse(self, src: List[str], proto: PrototypeMap):
        self._parse(src)
        self._declare(proto)


# ============================================================================
# 主程序入口
# ============================================================================

GOOS = {"linux", "darwin", "windows", "freebsd"}
GOARCH = {"amd64", "arm64", "riscv64"}


def make_subr_filename(name: str) -> str:
    name = os.path.basename(name)
    base = os.path.splitext(name)[0].rsplit("_", 2)
    
    if base[-1] in GOOS:
        return f"{'_'.join(base[:-1])}_subr_{base[-1]}.go"
    if base[-1] not in GOARCH:
        return f"{'_'.join(base)}_subr.go"
    if len(base) > 2 and base[-2] in GOOS:
        return f"{'_'.join(base[:-2])}_subr_{base[-2]}_{base[-1]}.go"
    return f"{'_'.join(base[:-1])}_subr_{base[-1]}.go"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LLVM RISC-V asm to Go asm.")
    parser.add_argument(
        "proto_file", type=str, help="The Go file that declares Go functions"
    )
    parser.add_argument(
        "asm_file", type=str, nargs="+", help="The LLVM assembly file(s)"
    )
    parser.add_argument(
        "-r",
        default=False,
        action="store_true",
        help="True: output as raw; default is False",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    
    # File handler
    log_dir = "asm2riscv_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, os.path.basename(os.path.splitext(args.proto_file)[0]) + ".log"
    )
    
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    log.setLevel(logging.DEBUG)

    log.info("Starting asm2asm conversion for RISC-V.")
    log.debug(f"Arguments: {args}")
    log.info(f"Detailed logs will be written to: {log_filename}")

    global OUTPUT_RAW
    if args.r:
        OUTPUT_RAW = True
        
    proto_name = os.path.splitext(args.proto_file)[0]

    # Parse prototype file
    log.info(f"Parsing prototype file: {args.proto_file}")
    try:
        with open(args.proto_file, "r", newline=None, encoding="utf-8") as fp:
            pkg, proto = PrototypeMap.parse(fp.read())
        log.info(f"Found {len(proto)} function prototypes in package '{pkg}'.")
    except Exception as e:
        log.error(f"Failed to parse prototype file: {e}")
        sys.exit(1)

    # Read assembly files
    src = []
    log.info(f"Reading assembly files: {args.asm_file}")
    try:
        for fn in args.asm_file:
            with open(fn, "r", newline=None, encoding="utf-8") as fp:
                src.extend(fp.read().splitlines())
        log.info(f"Read {len(src)} lines from {len(args.asm_file)} assembly file(s).")
    except Exception as e:
        log.error(f"Failed to read assembly files: {e}")
        sys.exit(1)

    # Initialize assembler
    asm = Assembler()

    # Generate header
    if OUTPUT_RAW:
        asm.out.append("// +build riscv64")
        asm.out.append("// Code generated by asm2asm, DO NOT EDIT.")
        asm.out.append(f"\npackage {pkg}\n")
        asm.out.append(f"var Text{STUB_NAME} = []byte{{")
    else:
        asm.out.append("// +build !noasm !appengine")
        asm.out.append("// Code generated by asm2asm, DO NOT EDIT.\n")
        asm.out.append('#include "go_asm.h"')
        asm.out.append('#include "funcdata.h"')
        asm.out.append('#include "textflag.h"\n')

    # Parse and convert
    try:
        asm.parse(src, proto)
    except Exception as e:
        log.error(f"Failed during parsing/conversion: {e}")
        sys.exit(1)

    # Write output assembly file
    asrc = proto_name + (".s" if not OUTPUT_RAW else "_text_riscv64.go")
    try:
        with open(asrc, "w", encoding="utf-8") as fp:
            for line in asm.out:
                print(line, file=fp)
            if OUTPUT_RAW:
                print("}", file=fp)
        log.info(f"Generated assembly file: {asrc}")
    except Exception as e:
        log.error(f"Failed to write assembly output file: {e}")
        sys.exit(1)

    # Generate subroutine file
    subr_file = os.path.join(
        os.path.dirname(args.proto_file), make_subr_filename(args.proto_file)
    )
    
    try:
        with open(subr_file, "w", encoding="utf-8") as fp:
            print("// +build !noasm !appengine", file=fp)
            print("// Code generated by asm2asm, DO NOT EDIT.\n", file=fp)
            print(f"package {pkg}\n", file=fp)
            
            if not asm.subr:
                log.warning("No subroutines found to generate.")
                return
            
            if OUTPUT_RAW:
                print("import (\n\t`github.com/bytedance/sonic/loader`\n)", file=fp)
                
                print("\nconst (", file=fp)
                for name in asm.code.funcs:
                    if (addr := asm.code.get(name)) is not None:
                        print(f"    _entry_{name} = {addr}", file=fp)
                print(")", file=fp)
                
                print("\nconst (", file=fp)
                for name in asm.code.funcs:
                    print(f"    _stack_{name} = {asm.code.stacksize(name)}", file=fp)
                print(")", file=fp)
                
                print("\nconst (", file=fp)
                for name, pcsp in asm.code.funcs.items():
                    if pcsp:
                        pcsp.optimize()
                        print(f"    _size_{name} = {pcsp.maxpc - pcsp.entry}", file=fp)
                print(")", file=fp)
                
                print("\nvar (", file=fp)
                for name, pcsp in asm.code.funcs.items():
                    if pcsp:
                        print(f"    _pcsp_{name} = {pcsp}", file=fp)
                print(")", file=fp)
                
                print("\nvar Funcs = []loader.CFunc{", file=fp)
                print(f'    {{"{STUB_NAME}", 0, {STUB_SIZE}, 0, nil}},', file=fp)
                for name in asm.code.funcs:
                    print(
                        f'    {{"{name}", _entry_{name}, _size_{name}, _stack_{name}, _pcsp_{name}}},',
                        file=fp,
                    )
                print("}", file=fp)
            else:
                print("//go:nosplit\n//go:noescape\n//goland:noinspection ALL", file=fp)
                for name in asm.subr:
                    print(f"func _{name}_entry__() uintptr", file=fp)
                
                print("\nvar (", file=fp)
                mlen = max((len(s) for s in asm.subr), default=0)
                for name, entry in asm.subr.items():
                    print(
                        f"    _subr_{name.ljust(mlen)} uintptr = _{name}_entry__() + {entry}",
                        file=fp,
                    )
                print(")", file=fp)
                
                print("\nconst (", file=fp)
                for name in asm.subr:
                    print(f"    _stack_{name} = {asm.code.stacksize(name)}", file=fp)
                print(")", file=fp)
                
                print("\nvar (", file=fp)
                for name in asm.subr:
                    print(f"    _ = _subr_{name}", file=fp)
                print(")", file=fp)
                
                print("\nconst (", file=fp)
                for name in asm.subr:
                    print(f"    _ = _stack_{name}", file=fp)
                print(")", file=fp)
                
        log.info(f"Generated Go subroutine file: {subr_file}")
        
    except Exception as e:
        log.error(f"Failed to write subroutine file: {e}")
        sys.exit(1)
        
    log.info("Conversion successful.")


if __name__ == "__main__":
    main()