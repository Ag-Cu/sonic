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
            try:
                return Token.num(int(ch, 8))
            except ValueError:
                return Token.num(int(ch)) # Not a valid octal
        return Token.num(int(ch))

    def _name(self, ch: str) -> Token:
        # 允许 '.' 出现在名称中 (除了第一个字符，由 _read 控制)
        while not self._eof and (self._ch == "_" or self._ch.isalnum() or self._ch == '.'):
            ch += self._rch()
        return Token.name(ch)

    def _read(self, ch: str) -> Token:
        if ch.isdigit():
            return self._int(ch)
        # 允许名称以 '.', '_' 或字母开头
        if ch.isalpha() or ch == '_' or ch == '.':
            return self._name(ch)
        if ch in ("*", "<", ">") and not self._eof and self._ch == ch:
            # 注意：这里原代码是 self._rch() * 2，这会消耗一个额外的字符。
            # 正确的方式应该是 ch + self._rch()
            return Token.punc(ch + self._rch())
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
        return 4

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
            if len(self.data) < 4:
                # This should not happen if we disable RVC
                log.error(f"Instruction data for '{self.asm_code}' is less than 4 bytes: {self.data.hex()}.")
                return f"// ERROR: UNEXPECTED 2-BYTE INSTRUCTION: {self.asm_code}"
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
        return self.func().to_bytes(self.len, "little", signed=True)

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
        if isinstance(self.instr.data, str):
            if not self.instr.data:
                return f"\t// {self.instr.asm_code} (skipped, covered by MOV address load)"
            return f"\t{self.instr.data}  // {self.instr.asm_code}"
        
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
        clean_name = self.instr.lstrip('.')
        return f"{clean_name}:"


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
        size = self.size(pc)
        if size == 0:
            return ""
        buf = bytes([self.fill]) * size
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
        pc = 0
        for block in self.blocks:
            for instr in block.body:
                if isinstance(instr, LabelInstr) and instr.instr == name:
                    return pc
                pc += instr.size(pc)
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

    # --- 已修改：添加日志 ---
    def _trace_block(self, bb: BasicBlock, pcsp: Optional[Pcsp]) -> int:
        """
        追踪基本块的栈深度，带有缓存和递归检测。
        """
        log.debug(f"进入对基本块 '{bb.name}' 的栈追踪...")
        if pcsp is not None:
            if bb.name in self.funcs:
                pcsp = None
            else:
                pcsp.pc = self.get(bb.name)
                if bb.func or pcsp.pc < pcsp.entry:
                    pcsp = Pcsp(pcsp.pc)
                    self.funcs[bb.name] = pcsp
        
        # 检查缓存或递归状态
        if bb.maxsp == -1:
            # -1: 从未访问，继续进行完整追踪
            return self._trace_nocache(bb, pcsp)
        if bb.maxsp >= 0:
            # >= 0: 已计算过，直接返回缓存结果
            log.debug(f"  块 '{bb.name}' 缓存命中。返回缓存的最大SP: {bb.maxsp}")
            return bb.maxsp
        if bb.maxsp == -2:
            # -2: 正在追踪中（递归），返回0以打破循环
            log.debug(f"  检测到块 '{bb.name}' 的递归调用。在此路径上视为SP变化为0。")
            return 0
        return 0 # 理论上不应到达这里

    # --- 已修改：添加日志 ---
    def _trace_nocache(self, bb: BasicBlock, pcsp: Optional[Pcsp]) -> int:
        """
        对一个尚未缓存的基本块进行完整的栈深度追踪。
        """
        log.debug(f"开始对基本块 '{bb.name}' 进行无缓存的栈追踪...")
        bb.maxsp = -2  # 标记为“正在追踪”以防止无限递归
        
        if pcsp:
            pc0, sp0 = pcsp.pc, pcsp.sp
            
        # 1. 计算当前块内的最大栈深度
        maxsp, term = self._trace_instructions(bb, pcsp)
        
        if term:
            log.debug(f"块 '{bb.name}' 是一个终止块。块内最大SP: {maxsp}")
            bb.maxsp = maxsp # 缓存结果
            return maxsp
            
        # 2. 递归追踪后续块
        a, b = 0, 0
        if pcsp:
            pc, sp = pcsp.pc, pcsp.sp
            
        if bb.jump:
            log.debug(f"  ...递归到跳转目标 '{bb.jump.name}'")
            a = self._trace_block(bb.jump, pcsp)
            if pcsp:
                pcsp.pc, pcsp.sp = pc, sp
                
        if bb.next:
            log.debug(f"  ...递归到下一个顺序块 '{bb.next.name}'")
            b = self._trace_block(bb.next, pcsp)
            if pcsp:
                pcsp.pc, pcsp.sp = pc, sp
                
        if pcsp:
            pcsp.pc, pcsp.sp = pc0, sp0
            
        # 3. 合并结果并缓存
        bb.maxsp = maxsp + max(a, b)
        log.debug(f"完成追踪基本块 '{bb.name}'。累计最大SP: {bb.maxsp}")
        return bb.maxsp

    # --- 已修改：添加日志 ---
    def _trace_instructions(self, bb: BasicBlock, pcsp: Optional[Pcsp]) -> Tuple[int, bool]:
        """
        遍历基本块中的所有指令，计算局部栈指针变化。
        """
        cursp, maxsp, close = 0, 0, False
        log.debug(f"  正在分析 '{bb.name}' 中的指令...")
        
        for ins in bb.body:
            diff = 0
            if isinstance(ins, RVInstr):
                instr_obj = ins.instr
                if instr_obj.is_return:
                    close = True
                # 关键：识别修改sp的addi指令
                if instr_obj.mnemonic == "addi" and "sp" in instr_obj.asm_code:
                    parts = instr_obj.asm_code.replace(",", " ").split()
                    # 确保是 addi sp, sp, imm 的形式
                    if len(parts) >= 4 and parts[1] == "sp" and parts[2] == "sp":
                        try:
                            # 汇编中 addi sp, sp, -16 是分配栈空间，所以 diff 为正
                            diff = -int(parts[3])
                        except (ValueError, IndexError):
                            pass
                
                if diff != 0:
                    log.debug(f"    指令 '{instr_obj.asm_code}' 改变SP {diff}。块内新SP偏移: {cursp + diff}")

                cursp += diff
                if cursp > maxsp:
                    log.debug(f"    块内最大SP偏移更新为: {cursp}")
                    maxsp = cursp
                    
            if pcsp:
                pcsp.update(ins.size(pcsp.pc), diff)
                
        log.debug(f"  '{bb.name}' 指令分析完毕。块内最大SP: {maxsp}, 是否为终止块: {close}")
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

    # --- 已修改：添加日志 ---
    def stacksize(self, name: str) -> int:
        """
        计算指定函数的栈大小。这是栈分析的入口点。
        """
        if name not in self.labels:
            raise SyntaxError("undefined function: " + name)
        
        log.info(f"正在为函数 '{name}' 计算栈大小...")
        size = self._trace_block(self.labels[name], None)
        log.info(f"函数 '{name}' 的计算栈大小为 {size} 字节。")
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
    "a0": ("MOV", "X10"), "a1": ("MOV", "X11"), "a2": ("MOV", "X12"), "a3": ("MOV", "X13"),
    "a4": ("MOV", "X14"), "a5": ("MOV", "X15"), "a6": ("MOV", "X16"), "a7": ("MOV", "X17"),
    "fa0": ("MOVD", "F10"), "fa1": ("MOVD", "F11"), "fa2": ("MOVD", "F12"), "fa3": ("MOVD", "F13"),
    "fa4": ("MOVD", "F14"), "fa5": ("MOVD", "F15"), "fa6": ("MOVD", "F16"), "fa7": ("MOVD", "F17"),
}

GNU_TO_GO_REG = {
    "zero": "ZERO", "ra": "X1", "sp": "X2", "gp": "X3", "tp": "X4",
    "t0": "X5", "t1": "X6", "t2": "X7", "s0": "X8", "fp": "X8", "s1": "X9",
    "a0": "X10", "a1": "X11", "a2": "X12", "a3": "X13", "a4": "X14",
    "a5": "X15", "a6": "X16", "a7": "X17",
    "s2": "X18", "s3": "X19", "s4": "X20", "s5": "X21", "s6": "X22",
    "s7": "X23", "s8": "X24", "s9": "X25", "s10": "X26", "s11": "g",
    "t3": "X28", "t4": "X29", "t5": "X30", "t6": "X31",
}

for i in range(32):
    GNU_TO_GO_REG[f"x{i}"] = GNU_TO_GO_REG.get({
        0: "zero", 1: "ra", 2: "sp", 3: "gp", 4: "tp", 8: "s0", 9: "s1", 27: "s11"
    }.get(i), f"X{i}")

for i in range(32):
    GNU_TO_GO_REG[f"f{i}"] = f"F{i}"
    GNU_TO_GO_REG[f"ft{i}"] = f"F{i}"
    GNU_TO_GO_REG[f"fs{i}"] = f"F{i}"
    GNU_TO_GO_REG[f"fa{i}"] = f"F{i}"

STUB_NAME = "native_entry"
STUB_SIZE = 67
WITH_OFFS = os.getenv("ASM2ASM_DEBUG_OFFSET", "").lower() in ("1", "yes", "true")
OUTPUT_RAW = False


class Assembler:
    """RISC-V汇编器"""

    out: List[str]
    data_out: List[str]
    subr: Dict[str, int]
    code: CodeSection
    vals: Dict[str, Union[str, int]]
    pending_lui: Dict[str, str]
    entry_point_name: Optional[str]
    current_section: str
    symbol_sizes: Dict[str, int]
    current_data_symbol: Optional[str]
    data_buffer: List[Tuple[str, int]]
    pkg_name: str
    entry_points: set[str]
    symbol_prefix: Optional[str]

    def __init__(self, pkg_name: str):
        self.out = []
        self.data_out = []
        self.subr = {}
        self.vals = {}
        self.code = CodeSection()
        self.pending_lui = {}
        # self.entry_point_name = None
        self.entry_points = set()
        self.current_section = 'text'
        self.symbol_sizes = {}
        self.current_data_symbol = None
        self.data_buffer = []
        self.pkg_name = pkg_name
        self.functions: Dict[str, CodeSection] = {} # 存储每个函数的 CodeSection
        self.current_function: Optional[str] = None # 当前正在解析的函数名
        self.code: Optional[CodeSection] = None # 指向当前函数的 CodeSection

        self.global_data: Dict[str, Dict[str, Any]] = {}
        self.symbol_prefix = None

        log.info("Assembler initialized.")

    @functools.cached_property
    def _symbolic_handlers(self):
        return {
            # 指令: (处理函数, 是否需要传递助记符)
            'lui': (self._handle_lui_hi, False),
            'addi': (self._handle_addi_lo, False),

            'ld': (self._handle_load_lo, True),
            'lw': (self._handle_load_lo, True),
            'lh': (self._handle_load_lo, True),
            'lb': (self._handle_load_lo, True),
            'lwu': (self._handle_load_lo, True),
            'lhu': (self._handle_load_lo, True),
            'lbu': (self._handle_load_lo, True),
            'fld': (self._handle_load_lo, True),
            'flw': (self._handle_load_lo, True),

            'beq': (self._handle_branch_rs_rs_label, True),
            'bne': (self._handle_branch_rs_rs_label, True),
            'blt': (self._handle_branch_rs_rs_label, True),
            'bge': (self._handle_branch_rs_rs_label, True),
            'bltu': (self._handle_branch_rs_rs_label, True),
            'bgeu': (self._handle_branch_rs_rs_label, True),
            'beqz': (self._handle_branch_rs_label, True),
            'bnez': (self._handle_branch_rs_label, True),
            'bltz': (self._handle_branch_rs_label, True),
            'bgez': (self._handle_branch_rs_label, True),
            'blez': (self._handle_branch_rs_label, True),
            'bgtz': (self._handle_branch_rs_label, True),
            'j': (self._handle_jump_label, True),
            'call': (self._handle_call_symbol, True),
            'tail': (self._handle_call_symbol, True),
            'ret': (self._handle_ret, True),
        }
    
    def _get_go_symbol_name(self, original_name: str) -> str:
        if not self.symbol_prefix:
            raise RuntimeError("Symbol prefix was not initialized before use.")
            
        clean_name = original_name.replace('.', '_').lstrip('_')
        return f"{self.symbol_prefix}_{clean_name}"

    def _get(self, v: str) -> int:
        if v in self.symbol_sizes:
            return self.symbol_sizes[v]
        if v not in self.vals:
            return self.code.get(v)
        if isinstance(self.vals[v], int):
            return self.vals[v]
        ret = self.vals[v] = self._eval(self.vals[v])
        return ret

    def _eval(self, v: str) -> int:
        return Expression(v).eval(self._get)

    def _emit_data(self, directive: str, values: List[str]):
        if not self.current_data_symbol:
            raise SyntaxError(f"Data directive '{directive}' used outside of a labeled data symbol.")
        
        for val_str in values:
            # --- 修正点 ---
            # 增加了对各种数据类型的处理
            if directive == ".ascii" or directive == ".asciz":
                # .ascii/.asciz 的处理逻辑保持不变
                data_bytes = val_str.encode('latin-1')
                if directive == ".asciz":
                    data_bytes += b'\0'
                for byte in data_bytes:
                    self.data_buffer.append(('.byte', byte))
            elif directive == ".byte":
                val = self._eval(val_str)
                self.data_buffer.append(('.byte', val))
            elif directive == ".quad":
                val = self._eval(val_str)
                self.data_buffer.append(('.quad', val))
            elif directive in (".word", ".long", ".int"):
                val = self._eval(val_str)
                self.data_buffer.append(('.word', val))
            elif directive in (".hword", ".short"):
                val = self._eval(val_str)
                self.data_buffer.append(('.hword', val))
            else:
                log.warning(f"Unsupported data directive in data section: {directive}")

    def _flush_data_buffer(self):
        if not self.current_data_symbol or not self.data_buffer:
            return

        size = self.symbol_sizes.get(self.current_data_symbol, 0)
        self.global_data[self.current_data_symbol] = {
            "buffer": self.data_buffer.copy(),
            "size": size
        }
        log.debug(f"Buffered data for symbol '{self.current_data_symbol}' with {len(self.data_buffer)} items.")

        self.data_buffer = []
        self.current_data_symbol = None

    # 替换 Assembler._generate_data_assembly 方法

    def _generate_data_assembly(self):
        log.info("Generating assembly for global data symbols...")
        if not self.global_data:
            log.info("No global data symbols found to generate.")
            return

        for symbol, data_info in self.global_data.items():
            buffer = data_info["buffer"]
            size = data_info["size"]
            
            go_symbol_name = self._get_go_symbol_name(symbol)
            
            calculated_size = 0
            for dtype, _ in buffer:
                if dtype == '.byte': calculated_size += 1
                elif dtype in ('.hword', '.short'): calculated_size += 2
                elif dtype in ('.word', '.long', '.int'): calculated_size += 4
                elif dtype == '.quad': calculated_size += 8
            
            if size == 0:
                size = calculated_size
            elif size != calculated_size and calculated_size > 0:
                log.warning(f"Size mismatch for symbol {symbol}: .size says {size}, calculated {calculated_size}. Using calculated size.")
                size = calculated_size

            if size == 0 and calculated_size == 0:
                log.debug(f"Skipping empty data symbol: {symbol}")
                continue

            self.data_out.append(f"GLOBL ·{go_symbol_name}(SB), RODATA, ${size}")
            
            offset = 0
            byte_stream = []

            # 将所有数据指令转换为字节流
            for dtype, val in buffer:
                if dtype == '.byte':
                    # 对于 byte，我们需要确保它在 0-255 范围内，如果是负数则取其补码
                    byte_stream.append(val & 0xFF)
                elif dtype in ('.hword', '.short'):
                    byte_stream.extend(val.to_bytes(2, 'little', signed=True)) # <--- 添加 signed=True
                elif dtype in ('.word', '.long', '.int'):
                    byte_stream.extend(val.to_bytes(4, 'little', signed=True)) # <--- 添加 signed=True
                elif dtype == '.quad':
                    byte_stream.extend(val.to_bytes(8, 'little', signed=True)) # <--- 添加 signed=True
            
            # 按8字节分块生成DATA指令
            for i in range(0, len(byte_stream), 8):
                chunk = byte_stream[i:i+8]
                # 如果最后一块不足8字节，用0填充
                if len(chunk) < 8:
                    chunk.extend([0] * (8 - len(chunk)))
                
                # 将8字节块解包为两个32位整数，然后格式化为Go汇编
                w1 = struct.unpack('<I', bytes(chunk[:4]))[0]
                w2 = struct.unpack('<I', bytes(chunk[4:]))[0]
                self.data_out.append(f"DATA ·{go_symbol_name}+{offset}(SB)/8, $0x{w2:08x}{w1:08x}")
                offset += 8

        log.info("Finished generating data assembly.")

    def _limit(self, v: int, a: int, b: int) -> int:
        if not (a <= v <= b):
            raise SyntaxError(f"integer constant out of bound [{a}, {b}): {v}")
        return v

    def _assemble_instruction(self, ins: str) -> bytes:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.s', delete=False, encoding='utf-8') as asm_file:
                asm_file.write(f".text\n.align 2\n{ins}\n")
                asm_file_name = asm_file.name
            
            # Add -mattr=-c to disable compressed instructions
            result = subprocess.run([
                'llvm-mc', '-arch=riscv64', '-mattr=+m,+a,+f,+d,+v,-c', '--show-encoding', asm_file_name
            ], capture_output=True, text=True, timeout=5, check=True)
            
            hex_pattern = r'# encoding: \[(0x[0-9a-f,x\s]+)\]'
            match = re.search(hex_pattern, result.stdout)
            if match:
                hex_str = match.group(1)
                hex_bytes = [int(x.strip(), 16) for x in hex_str.split(',') if x.strip()]
                return bytes(hex_bytes)
            
            raise ValueError(f"Could not extract encoding from llvm-mc output for '{ins}':\n{result.stdout}\n{result.stderr}")
                
        except subprocess.CalledProcessError as e:
            log.error(f"llvm-mc failed for '{ins}':\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            raise
        finally:
            if 'asm_file_name' in locals() and os.path.exists(asm_file_name):
                os.unlink(asm_file_name)

    def _translate_reg(self, reg: str) -> str:
        return GNU_TO_GO_REG.get(reg.strip().lower(), reg.upper())

    def _get_go_branch_mnemonic(self, mnemonic: str, r1: str, r2: str) -> Tuple[str, str, str]:
        mapping = {
            'beq': 'BEQ', 'bne': 'BNE', 'blt': 'BLT', 'bge': 'BGE',
            'bltu': 'BLTU', 'bgeu': 'BGEU',
            'beqz': ('BEQ', r1, 'ZERO'),
            'bnez': ('BNE', r1, 'ZERO'),
            'blez': ('BGE', 'ZERO', r1), # b le z <=> 0 ge r1
            'bgez': ('BGE', r1, 'ZERO'),
            'bltz': ('BLT', r1, 'ZERO'),
            'bgtz': ('BLT', 'ZERO', r1), # b gt z <=> 0 lt r1
        }
        if mnemonic in mapping:
            val = mapping[mnemonic]
            if isinstance(val, tuple):
                return val[0], self._translate_reg(val[1]), self._translate_reg(val[2])
            return val, self._translate_reg(r1), self._translate_reg(r2)
        return mnemonic.upper(), self._translate_reg(r1), self._translate_reg(r2)

    def _try_pattern_match(self, clean_line: str) -> Optional[str]:
        """
        Dispatch to a specific handler based on the instruction mnemonic.
        This avoids complex regex and improves maintainability.
        """
        parts = clean_line.replace(',', ' ').split()
        if not parts:
            return None

        mnemonic = parts[0].lower()
        operands = [p.strip() for p in parts[1:] if p]

        handler_info = self._symbolic_handlers.get(mnemonic)
        if handler_info:
            handler, pass_mnemonic = handler_info
            if pass_mnemonic:
                return handler(mnemonic, operands)
            else:
                return handler(operands)
        
        return None

    def _process_instruction(self, line: str) -> Instruction:
        instr = Instruction(line)
        clean_line = line.strip()
        
        if instr.is_branch:
            parts = clean_line.split(None, 1)
            if len(parts) > 1:
                operands = parts[1].split(',')
                instr.label_name = operands[-1].strip().lstrip('.')
        
        if (translated := self._try_pattern_match(clean_line)) is not None:
            instr.data = translated
            log.debug(f"Symbolically translated '{line}' -> '{translated}'")
            return instr
        
        try:
            binary_data = self._assemble_instruction(line)
            instr.data = binary_data
        except Exception as e:
            log.warning(f"Failed to assemble '{line}', outputting NOP. Error: {e}")
            instr.data = b'\x13\x00\x00\x00'
        
        return instr

    def _cmd_nop(self, _: List[str]): pass
    def _cmd_set(self, args: List[str]):
        if len(args) != 2: raise SyntaxError(".set takes 2 arguments")
        if not args[0].isidentifier(): raise SyntaxError(f"{repr(args[0])} is not a valid identifier")
        self.vals[args[0]] = args[1]
        log.debug(f"Handled .set {args[0]} = {args[1]}")

    def _cmd_byte(self, args: List[str]):
        if self.current_section != 'text': self._emit_data(".byte", args)
        else: self.code.lazy(1, lambda: self._limit(self._eval(args[0]), -0x80, 0xFF) & 0xFF, f".byte {args[0]}")

    def _cmd_word(self, args: List[str]):
        if self.current_section != 'text': self._emit_data(".word", args)
        else: self.code.lazy(2, lambda: self._limit(self._eval(args[0]), -0x8000, 0xFFFF) & 0xFFFF, f".word {args[0]}")

    def _cmd_long(self, args: List[str]):
        if self.current_section != 'text': self._emit_data(".long", args)
        else: self.code.lazy(4, lambda: self._limit(self._eval(args[0]), -0x80000000, 0xFFFFFFFF) & 0xFFFFFFFF, f".long {args[0]}")

    def _cmd_quad(self, args: List[str]):
        if self.current_section != 'text': self._emit_data(".quad", args)
        else: self.code.lazy(8, lambda: self._limit(self._eval(args[0]), -0x8000000000000000, 0xFFFFFFFFFFFFFFFF) & 0xFFFFFFFFFFFFFFFF, f".quad {args[0]}")

    def _cmd_ascii(self, args: List[str]):
        if len(args) != 1: raise SyntaxError(".ascii takes 1 argument")
        if self.current_section != 'text': self._emit_data(".ascii", args)
        else: self.code.emit(args[0].encode("latin-1"), ".ascii")

    def _cmd_asciz(self, args: List[str]):
        if len(args) != 1: raise SyntaxError(".asciz takes 1 argument")
        if self.current_section != 'text': self._emit_data(".asciz", args)
        else: self.code.emit(args[0].encode("latin-1") + b"\0", ".asciz")

    def _cmd_space(self, args: List[str]):
        nb = self._eval(args[0])
        fv = self._limit(self._eval(args[1]), 0, 255) if len(args) > 1 else 0
        if self.current_section != 'text': log.warning(".space in data section is not fully supported, skipping.")
        else: self.code.emit(bytes([fv] * nb), ".space")
    
    def _cmd_zero(self, args: List[str]):
        # --- 修正点 ---
        # 允许 .zero 接受 1 或 2 个参数
        if not (1 <= len(args) <= 2):
            raise SyntaxError(".zero takes 1 or 2 arguments")
            
        size = self._eval(args[0])
        # 如果有第二个参数，则使用它作为填充值；否则默认为 0
        fill_val = self._eval(args[1]) if len(args) > 1 else 0

        if self.current_section == 'data':
            if not self.current_data_symbol:
                raise SyntaxError(".zero used in data section outside of a labeled symbol.")
            for _ in range(size):
                self.data_buffer.append(('.byte', fill_val))
        else: # text section
            self.code.emit(bytes([fill_val] * size), f".zero {', '.join(args)}")

    def _cmd_section(self, args: List[str]):
        self._flush_data_buffer()
        section_name = args[0].split(',')[0].strip().strip('"')
        if section_name == '.text':
            self.current_section = 'text'
        elif '.rodata' in section_name or '.data' in section_name or '.srodata' in section_name:
            self.current_section = 'data'
        else:
            self.current_section = 'other'
        log.debug(f"Switched to section: {section_name} (type: {self.current_section})")

    def _cmd_size(self, args: List[str]):
        if len(args) == 2:
            symbol, size_expr = args[0], args[1]
            try:
                if '-' in size_expr:
                    end, start = size_expr.split('-')
                    size = self.code.get(end) - self.code.get(start)
                else:
                    size = self._eval(size_expr)
                self.symbol_sizes[symbol] = size
                log.debug(f"Recorded size for symbol '{symbol}': {size}")
            except Exception as e:
                log.warning(f"Could not evaluate .size expression '{size_expr}': {e}")

    @functools.cached_property
    def _commands(self) -> dict:
        return {
            # 保留需要处理的指令
            ".set": self._cmd_set,
            ".byte": self._cmd_byte,
            ".word": self._cmd_word,
            ".hword": self._cmd_word,
            ".short": self._cmd_word,
            ".int": self._cmd_long,
            ".long": self._cmd_long,
            ".quad": self._cmd_quad,
            ".ascii": self._cmd_ascii,
            ".asciz": self._cmd_asciz,
            ".space": self._cmd_space,
            ".zero": self._cmd_zero,
            
            # 将可以忽略的指令全部指向 _cmd_nop
            ".p2align": self._cmd_nop,
            ".align": self._cmd_nop,
            ".globl": self._cmd_nop,
            ".text": self._cmd_nop,  # .text 在 _parse 中有特殊处理，这里设为 nop 无妨
            ".file": self._cmd_nop,
            ".type": self._cmd_nop,  # .type 在 _parse 中有特殊处理
            ".size": self._cmd_nop,
            ".section": self._cmd_nop, # .section 在 _parse 中有特殊处理
            ".attribute": self._cmd_nop,
            ".ident": self._cmd_nop,
            ".addrsig": self._cmd_nop,
        }

    @staticmethod
    def _remove_comments(line: str) -> str:
        return line.split("//")[0].split("#")[0].split(';')[0]

        # 在 Assembler 类中

    def _parse(self, src: List[str]):
        log.info("Starting assembly parsing phase...")
        # last_globl = None
        
        for i, line in enumerate(src):
            line = self._remove_comments(line).strip()
            if not line:
                continue
                
            log.debug(f"Parsing line {i+1}: '{line}'")
            
            if line.lower().startswith(".globl"):
                continue

            # .type 是定义函数的权威来源
            if line.lower().startswith(".type"):
                parts = line.replace(',', ' ').split()
                if len(parts) == 3 and parts[0].lower() == '.type' and parts[2] == '@function':
                    symbol = parts[1]
                    self.current_function = symbol
                    self.functions[symbol] = CodeSection()
                    self.code = self.functions[symbol]
                    log.info(f"Found function definition: '{symbol}'")
                    self.current_section = 'text'
                    log.debug(f"Switched to section: .text (triggered by .type @function for '{symbol}')")
                continue

            if line.endswith(":"):
                label_name = line[:-1]
                if self.current_section == 'text' and self.code:
                    self.code.label(label_name)
                else:
                    self._flush_data_buffer()
                    self.current_data_symbol = label_name
                    log.debug(f"Defining data symbol: '{label_name}'")
                continue
                
            if line.startswith("."):
                cmd = Command.parse(line)
                # 特殊处理会改变状态的伪指令
                if cmd.cmd.lower() == '.text':
                    self._flush_data_buffer()
                    self.current_function = None
                    self.code = None
                    self.current_section = 'text'
                    log.debug("Switched to section: .text")
                    continue
                elif cmd.cmd.lower() == '.section':
                    self._cmd_section(cmd.args) # 仍然需要调用它来切换状态
                    continue

                # 其他伪指令通过字典处理 (大部分是 nop)
                if func := self._commands.get(cmd.cmd):
                    func(cmd.args)
                else:
                    log.warning(f"Ignoring unknown directive: {cmd.cmd}")
                continue
                
            if self.current_section == 'text':
                if self.code:
                    instr = self._process_instruction(line)
                    if instr.data is not None:
                        self.code.instr(instr)
                else:
                    log.warning(f"Instruction '{line}' found in .text section but outside a function context, ignoring.")
            # 数据段中的指令已经被忽略，这里不需要 else
        
        self._flush_data_buffer()
        log.info("Assembly parsing finished.")

    def _reloc(self, rip: int = 0):
        log.info("Performing relocation pass (calculating PC for each instruction)...")
        for block in self.code.blocks:
            for instr in block.body:
                rip += instr.size(rip)
        log.info("Relocation pass finished.")

    def _declare(self, protos: PrototypeMap):
        log.info("Starting declaration and code generation phase...")
        
        if not self.functions:
            raise RuntimeError("No functions found in the assembly file.")
        
        # 为每个函数生成 TEXT 块
        for name, code_section in self.functions.items():
            self._declare_body(name, code_section)

        # 为 Go 原型中声明的函数生成包装器
        self._declare_functions(protos)
        log.info("Declaration and code generation finished.")

    def _declare_body(self, asm_name: str, code_section: CodeSection):
        size = code_section.stacksize(asm_name)
        
        # Conditionally name the TEXT block.
        if asm_name in self.entry_points:
            # This is a public entry point, give it the special name.
            go_entry_name = f"·__{asm_name}_riscv64_entry__(SB)"
            log.info(f"Generating public TEXT block for entry point: {go_entry_name}")
        else:
            # This is an internal static function, give it a simple name.
            prefixed_name = self._get_go_symbol_name(asm_name)
            go_entry_name = f"·{prefixed_name}(SB)"
            log.info(f"Generating internal TEXT block for static function: {go_entry_name}")

        self.out.append(f"TEXT {go_entry_name}, NOSPLIT, ${size}")
        self.out.append("\tNO_LOCAL_POINTERS")
        
        pc = 0
        for v in code_section.instrs:
            formatted_line = v.formatted(pc)
            if formatted_line.strip():
                self.out.append(formatted_line)
            pc += v.size(pc)
        self.out.append("")

    def _declare_function(self, name: str, proto: Prototype):
        asm_name = name.lstrip("_")
        log.info(f"Generating wrapper for function '·{name}(SB)'.")
        
        # 从 self.functions 字典中查找函数，而不是 self.code
        if asm_name not in self.functions:
            log.warning(f"Function '{asm_name}' not found in assembly, skipping wrapper.")
            return

        # 获取对应函数的 CodeSection
        code_section = self.functions[asm_name]
        
        # 对于Go包装器来说，原生函数的地址偏移量总是0，因为它调用的是原生函数的TEXT入口
        self.subr[asm_name] = 0
        
        # 1. 获取计算出的栈大小
        calculated_size = code_section.stacksize(asm_name)
        
        # 2. 参照ARM脚本，增加一个固定的最小/额外栈空间（例如64字节）
        #    这确保了即使 calculated_size 为 0，我们仍然有栈帧用于栈检查和运行时交互。
        wrapper_stack_size = calculated_size + 64

        log.debug(f"  - Calculated stack size: {calculated_size}, Arg space: {proto.argspace}, Wrapper stack size: {wrapper_stack_size}")
        
        # 3. Go 函数的 TEXT 声明，栈帧大小为0，因为栈在原生代码中管理
        self.out.append("")
        self.out.append(f"TEXT ·{name}(SB), NOSPLIT, $0-{proto.argspace}")
        self.out.append("\tNO_LOCAL_POINTERS")
        
        # 4. 栈检查逻辑 (总是生成)
        self.out.append(f"")
        self.out.append(f"_entry_{name}:")
        self.out.append("\tMOV 16(g), X31      // g.stack.hi")
        
        # 使用 wrapper_stack_size 进行检查
        if wrapper_stack_size < 2048:
            self.out.append(f"\tADD $-{wrapper_stack_size}, SP, X30")
        else:
            self.out.append(f"\tMOV $-{wrapper_stack_size}, X30")
            self.out.append("\tADD SP, X30, X30")
            
        self.out.append(f"\tBLTU X30, X31, _stack_grow_{name}")

        # 5. 函数体开始
        self.out.append(f"")
        self.out.append(f"_{name}:")

        # 加载参数到 RISC-V 调用约定的寄存器中
        offs = 0
        for arg in proto.args:
            op, reg = REG_MAP[arg.creg.reg]
            self.out.append(f"\t{op} {arg.name}+{offs}(FP), {reg}")
            offs += arg.size
        
        # 调用原生代码的入口点
        go_entry_name = f"·__{asm_name}_riscv64_entry__(SB)"
        self.out.append(f"\tCALL {go_entry_name}")
        
        # 处理返回值
        if proto.retv is not None:
            op, reg = REG_MAP[proto.retv.creg.reg]
            self.out.append(f"\t{op} {reg}, {proto.retv.name}+{offs}(FP)")
        
        self.out.append("\tRET")
        
        # 6. 栈增长的实现 (总是生成)
        self.out.append(f"")
        self.out.append(f"_stack_grow_{name}:")
        self.out.append("\tMOV X1, X3      // Save return address (RA)")
        self.out.append("\tCALL runtime·morestack_noctxt(SB)")
        self.out.append(f"\tJMP _entry_{name}")

    def _declare_functions(self, protos: PrototypeMap):
        log.info(f"Generating Go wrappers for {len(protos)} entry points.")
        for name, proto in sorted(protos.items()):
            if name.startswith("_"):
                self._declare_function(name, proto)
            else:
                raise SyntaxError('function prototype must have a "_" prefix: ' + repr(name))

    def parse(self, src: List[str], protos: PrototypeMap):
        # Populate the set of entry points from the Go prototype file.
        # The names in protos start with '_', so we strip it.
        self.entry_points = {name.lstrip('_') for name in protos.keys()}
        log.info(f"Identified {len(self.entry_points)} entry points from Go prototype: {self.entry_points}")

        if not protos:
            log.warning("Go prototype file is empty. Falling back to package name for symbol prefix.")
            self.symbol_prefix = self.pkg_name
        else:
            chosen_name = sorted(protos.keys())[0]
            prefix_hash = f"p{abs(hash(chosen_name)):x}"
            self.symbol_prefix = prefix_hash
            log.info(f"Using function '{chosen_name}' to generate symbol prefix: '{self.symbol_prefix}'")

        self._parse(src)
        self._generate_data_assembly()

    def _handle_lui_hi(self, operands: List[str]) -> str:
        """处理 lui reg, %hi(symbol)"""
        if len(operands) == 2 and operands[1].lower().startswith('%hi('):
            reg = operands[0]
            symbol = operands[1][4:-1]
            self.pending_lui[reg] = symbol
            go_symbol = self._get_go_symbol_name(symbol)
            return f"MOV $·{go_symbol}(SB), {self._translate_reg(reg)}"
        return None

    def _handle_addi_lo(self, operands: List[str]) -> str:
        """处理 addi rd, rs1, %lo(symbol)"""
        if len(operands) == 3 and operands[0] == operands[1] and operands[2].lower().startswith('%lo('):
            reg = operands[0]
            symbol = operands[2][4:-1]
            if self.pending_lui.get(reg) == symbol:
                del self.pending_lui[reg]
                return ""  # 已被 LUI 处理，跳过此条指令
        return None
    
    def _handle_load_lo(self, mnemonic: str, operands: List[str]) -> str:
        """处理 ld/fld rd, %lo(symbol)(rb) 等加载指令"""
        if len(operands) == 2:
            rd = operands[0]
            mem_op = operands[1]
            
            # 解析 offset(rb)
            match = re.match(r"%lo\((.+)\)\((.+)\)", mem_op, re.IGNORECASE)
            if match:
                symbol, rb = match.groups()
                if self.pending_lui.get(rb) == symbol:
                    del self.pending_lui[rb]
                    
                    go_mnemonic = "UNKNOWN"
                    mnemonic_lower = mnemonic.lower()
                    
                    if mnemonic_lower in ('ld'):
                        go_mnemonic = "MOV"
                    elif mnemonic_lower in ('fld'):
                        go_mnemonic = "MOVD"
                    else:
                        go_mnemonic = "UNKNOWN"

                    # LUI 已经将地址加载到 rb, 现在我们从该地址加载数据
                    return f"{go_mnemonic} ({self._translate_reg(rb)}), {self._translate_reg(rd)}"
        return None

    def _handle_branch_rs_rs_label(self, mnemonic: str, operands: List[str]) -> str:
        """处理 beq rs1, rs2, label 等双寄存器分支"""
        if len(operands) == 3:
            r1, r2, label = operands
            go_mnemonic = mnemonic.upper()
            return f"{go_mnemonic} {self._translate_reg(r1)}, {self._translate_reg(r2)}, {label.lstrip('.')}"
        return None

    def _handle_branch_rs_label(self, mnemonic: str, operands: List[str]) -> str:
        """处理 beqz rs, label 等单寄存器分支伪指令"""
        if len(operands) == 2:
            r1, label = operands
            go_mnemonic = mnemonic.upper()
            return f"{go_mnemonic} {self._translate_reg(r1)}, {label.lstrip('.')}"
        return None

    def _handle_jump_label(self, mnemonic: str, operands: List[str]) -> str:
        """处理 j label"""
        if len(operands) == 1:
            label = operands[0]
            return f"JMP {label.lstrip('.')}"
        return None

    def _handle_call_symbol(self, mnemonic: str, operands: List[str]) -> str:
        """处理 call/tail symbol"""
        if len(operands) == 1:
            symbol = operands[0]
            go_mnemonic = "CALL" if mnemonic.lower() == "call" else "JMP"
            is_public_entry = symbol.lstrip('_') in self.entry_points

            if is_public_entry:
                target_symbol = symbol
            else:
                target_symbol = self._get_go_symbol_name(symbol)

            return f"{go_mnemonic} ·{target_symbol.lstrip('.')}(SB)"
        return None

    def _handle_ret(self, mnemonic: str, operands: List[str]) -> str:
        """处理 ret"""
        return "RET"



# ============================================================================
# 主程序入口
# ============================================================================

GOOS = {"linux", "darwin", "windows", "freebsd"}
GOARCH = {"amd64", "arm64", "riscv64"}


def make_subr_filename(name: str) -> str:
    name = os.path.basename(name)
    base = os.path.splitext(name)[0].rsplit("_", 2)
    
    if base[-1] in GOOS: return f"{'_'.join(base[:-1])}_subr_{base[-1]}.go"
    if base[-1] not in GOARCH: return f"{'_'.join(base)}_subr.go"
    if len(base) > 2 and base[-2] in GOOS: return f"{'_'.join(base[:-2])}_subr_{base[-2]}_{base[-1]}.go"
    return f"{'_'.join(base[:-1])}_subr_{base[-1]}.go"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LLVM RISC-V asm to Go asm.")
    parser.add_argument("proto_file", type=str, help="The Go file that declares Go functions")
    parser.add_argument("asm_file", type=str, nargs="+", help="The LLVM assembly file(s)")
    parser.add_argument("-r", default=False, action="store_true", help="True: output as raw; default is False")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    
    log_dir = "asm2riscv_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, os.path.basename(os.path.splitext(args.proto_file)[0]) + ".log")
    
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    log.setLevel(logging.DEBUG)

    log.info("Starting asm2asm conversion for RISC-V.")
    log.debug(f"Arguments: {args}")
    log.info(f"Detailed logs will be written to: {log_filename}")

    global OUTPUT_RAW
    if args.r: OUTPUT_RAW = True
        
    proto_name = os.path.splitext(args.proto_file)[0]

    try:
        with open(args.proto_file, "r", newline=None, encoding="utf-8") as fp:
            pkg, proto = PrototypeMap.parse(fp.read())
        log.info(f"Found {len(proto)} function prototypes in package '{pkg}'.")
    except Exception as e:
        log.error(f"Failed to parse prototype file: {e}")
        sys.exit(1)

    src = []
    try:
        for fn in args.asm_file:
            with open(fn, "r", newline=None, encoding="utf-8") as fp:
                src.extend(fp.read().splitlines())
        log.info(f"Read {len(src)} lines from {len(args.asm_file)} assembly file(s).")
    except Exception as e:
        log.error(f"Failed to read assembly files: {e}")
        sys.exit(1)

    asm = Assembler(pkg)

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

    try:
        asm.parse(src, proto)

        asm._declare(proto)
        
    except Exception as e:
        log.error(f"Failed during parsing/conversion: {e}", exc_info=True)
        sys.exit(1)

    asrc = proto_name + (".s" if not OUTPUT_RAW else "_text_riscv64.go")
    try:
        with open(asrc, "w", encoding="utf-8") as fp:
            for line in asm.out:
                print(line, file=fp)
            
            if not OUTPUT_RAW and asm.data_out:
                print("\n// Data section", file=fp)
                for line in asm.data_out:
                    print(line, file=fp)

            if OUTPUT_RAW: print("}", file=fp)
        log.info(f"Generated assembly file: {asrc}")
    except Exception as e:
        log.error(f"Failed to write assembly output file: {e}")
        sys.exit(1)

    subr_file = os.path.join(os.path.dirname(args.proto_file), make_subr_filename(args.proto_file))
    
    try:
        with open(subr_file, "w", encoding="utf-8") as fp:
            print("// +build !noasm !appengine", file=fp)
            print("// Code generated by asm2asm, DO NOT EDIT.\n", file=fp)
            print(f"package {pkg}\n", file=fp)
            
            if not asm.subr:
                log.warning("No subroutines found to generate.")
                # Still create the file to satisfy build systems, but leave it empty.
                return
            
            if OUTPUT_RAW:
                # --- This block is also corrected for robustness ---
                print("import (\n\t`github.com/bytedance/sonic/loader`\n)", file=fp)
                print("\nconst (", file=fp)
                for name, code_section in asm.functions.items():
                    # The entry address within a function's own code section is always 0.
                    if (addr := code_section.get(name)) is not None: 
                        print(f"    _entry_{name} = {addr}", file=fp)
                print(")", file=fp)
                print("\nconst (", file=fp)
                for name, code_section in asm.functions.items(): 
                    print(f"    _stack_{name} = {code_section.stacksize(name)}", file=fp)
                print(")", file=fp)
                print("\nconst (", file=fp)
                for name, pcsp in asm.code.funcs.items(): # Note: This part might need review if pcsp logic is used
                    if pcsp:
                        pcsp.optimize()
                        print(f"    _size_{name} = {pcsp.maxpc - pcsp.entry}", file=fp)
                print(")", file=fp)
                print("\nvar (", file=fp)
                for name, pcsp in asm.code.funcs.items():
                    if pcsp: print(f"    _pcsp_{name} = {pcsp}", file=fp)
                print(")", file=fp)
                print("\nvar Funcs = []loader.CFunc{", file=fp)
                print(f'    {{"{STUB_NAME}", 0, {STUB_SIZE}, 0, nil}},', file=fp)
                for name in asm.functions:
                    print(f'    {{"{name}", _entry_{name}, _size_{name}, _stack_{name}, _pcsp_{name}}},', file=fp)
                print("}", file=fp)
            else:
                print("//go:nosplit\n//go:noescape\n//goland:noinspection ALL", file=fp)
                # --- MODIFIED ---
                # Declare the special entry point prototype for each wrapped function.
                for name in asm.subr:
                    print(f"func __{name}_riscv64_entry__() uintptr", file=fp)

                print("\nvar (", file=fp)
                mlen = max((len(s) for s in asm.subr), default=0)
                for name, entry in asm.subr.items():
                    # Reference the special entry point symbol.
                    symbol_to_call = f"__{name}_riscv64_entry__"
                    print(
                        f"    _subr__{name.ljust(mlen)} uintptr = {symbol_to_call}() + {entry}",
                        file=fp,
                    )
                print(")", file=fp)
                
                print("\nconst (", file=fp)
                for name in asm.subr: 
                    # Get the correct CodeSection for the function 'name'
                    code_section = asm.functions[name]
                    print(f"    _stack__{name} = {code_section.stacksize(name)}", file=fp)
                print(")", file=fp)

                print("\nvar (", file=fp)
                for name in asm.subr: print(f"    _ = _subr__{name}", file=fp)
                print(")", file=fp)

                print("\nconst (", file=fp)
                for name in asm.subr: print(f"    _ = _stack__{name}", file=fp)
                print(")", file=fp)
                
        log.info(f"Generated Go subroutine file: {subr_file}")
        
    except Exception as e:
        log.error(f"Failed to write subroutine file: {e}", exc_info=True) # Added exc_info for better debugging
        sys.exit(1)
        
    log.info("Conversion successful.")


if __name__ == "__main__":
    main()