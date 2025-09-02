#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import struct
import subprocess
import tempfile
import logging
import argparse
from typing import Optional, List, Dict, Tuple

from parsing import Line, Command, Expression
from prototype import Prototypes, CallingConvention
from subroutine import Subroutine, save_subr_refs, make_subr_filename

# --- Logging setup will be done in main ---

STUB_NAME = '__native_entry__'
ENTRY_SIZE = 12

def escape_bytes(b: bytes) -> str:
    """
    将字节序列转换为更易读的字符串表示，
    对不可打印字符和特殊字符进行转义。
    """
    res = []
    for byte in b:
        if 32 <= byte <= 126:
            char = chr(byte)
            if char == '"':
                res.append('\\"')
            elif char == '\\':
                res.append('\\\\')
            else:
                res.append(char)
        else:
            res.append(f'\\x{byte:02x}')
    return "".join(res)


class Label(str):
    @property
    def name(self) -> str:
        return self.replace('.', '_')
    def __str__(self) -> str:
        return self.name + ':'

class Instruction:
    text: str
    _stack_adj_re = re.compile(r'^\s*addi\s+sp,\s*sp,\s*(-?\d+)\s*$')
    _uncond_br_re = re.compile(r'^\s*(j|jal|tail|call)\s+.*$')
    _cond_br_re = re.compile(r'^\s*b(eq|ne|lt|ge|ltu|geu)z?\s+.*$')
    _ret_re = re.compile(r'^\s*ret\s*$')

    def __init__(self, text: str):
        self.text = text.strip()

    def __str__(self) -> str:
        return self.text

    @classmethod
    def parse(cls, ins: str) -> 'Instruction':
        logging.debug(f"正在解析原始指令文本: '{ins.strip()}'")
        return cls(ins.strip())

    @property
    def is_return(self) -> bool:
        return bool(self._ret_re.match(self.text))

    @property
    def is_unconditional_branch(self) -> bool:
        # jalr is often used for returns or indirect calls, treat as terminal for simple analysis
        if 'jalr' in self.text:
            return True
        return bool(self._uncond_br_re.match(self.text))

    @property
    def is_conditional_branch(self) -> bool:
        return bool(self._cond_br_re.match(self.text))

    @property
    def is_terminator(self) -> bool:
        """判断指令是否会终止一个基本块的线性执行流"""
        return self.is_return or self.is_unconditional_branch or self.is_conditional_branch

    @property
    def stack_adjustment(self) -> Optional[int]:
        """如果指令是 'addi sp, sp, imm', 返回 imm，否则返回 None"""
        match = self._stack_adj_re.match(self.text)
        if match:
            return int(match.group(1))
        return None

    def get_branch_target(self) -> Optional[str]:
        """获取分支或调用指令的目标标签"""
        parts = re.split(r'[\s,]+', self.text)
        if len(parts) > 1:
            # 目标通常是最后一个操作数
            target = parts[-1]
            # 简单的检查，看它是否像一个标签
            if re.match(r'^[._a-zA-Z][._a-zA-Z0-9]*$', target):
                return target.lstrip('.')
        return None

# --- 新增: BasicBlock 类 ---
class BasicBlock:
    name: str
    body: list
    next: Optional['BasicBlock']
    jump: Optional['BasicBlock']
    max_sp: int  # 用于栈分析的备忘录

    def __init__(self, name: str):
        self.name = name
        self.body = []
        self.next = None
        self.jump = None
        self.max_sp = -1 # -1: 未计算, -2: 正在计算 (防止无限递归)

    def __repr__(self):
        return f"BasicBlock('{self.name}')"

    def link_to(self, other: 'BasicBlock'):
        self.next = other

    def jump_to(self, other: 'BasicBlock'):
        self.jump = other

class Translator:
    out: list[str]
    mbuf: bytes
    subr: dict[str, Subroutine]
    labels: dict[str, int]
    relocations: dict[int, tuple]
    aliases: dict[str, str]
    comment_buf: list[str]
    sections: dict[str, list]
    data_sizes: dict[str, int]
    pending_lui: dict[str, str]

    # --- 新增: CFG 和栈分析相关的属性 ---
    blocks: List[BasicBlock]
    current_block: BasicBlock
    labels_to_blocks: Dict[str, BasicBlock]
    dead_code: bool # 标记当前是否处于不可达代码区域

    # --- 新增: RISC-V GNU/LLVM ABI 寄存器名到 Go 汇编名的映射 ---
    __gnu_to_go_reg_map = {
        'zero': 'ZERO', 'x0': 'ZERO',
        'ra': 'RA', 'x1': 'RA',
        'sp': 'SP', 'x2': 'SP',
        'gp': 'GP', 'x3': 'GP',
        'tp': 'TP', 'x4': 'TP',
        't0': 'X5', 'x5': 'X5',
        't1': 'X6', 'x6': 'X6',
        't2': 'X7', 'x7': 'X7',
        's0': 'X8', 'fp': 'X8', 'x8': 'X8',
        's1': 'X9', 'x9': 'X9',
        'a0': 'X10', 'x10': 'X10',
        'a1': 'X11', 'x11': 'X11',
        'a2': 'X12', 'x12': 'X12',
        'a3': 'X13', 'x13': 'X13',
        'a4': 'X14', 'x14': 'X14',
        'a5': 'X15', 'x15': 'X15',
        'a6': 'X16', 'x16': 'X16',
        'a7': 'X17', 'x17': 'X17',
        's2': 'X18', 'x18': 'X18',
        's3': 'X19', 'x19': 'X19',
        's4': 'X20', 'x20': 'X20',
        's5': 'X21', 'x21': 'X21',
        's6': 'X22', 'x22': 'X22',
        's7': 'X23', 'x23': 'X23',
        's8': 'X24', 'x24': 'X24',
        's9': 'X25', 'x25': 'X25',
        's10': 'X26', 'x26': 'X26',
        's11': 'g', 'x27': 'g',
        't3': 'X28', 'x28': 'X28',
        't4': 'X29', 'x29': 'X29',
        't5': 'X30', 'x30': 'X30',
        't6': 'X31', 'x31': 'X31',
    }

    def _translate_reg(self, reg_name: str) -> str:
        """翻译单个 GNU/LLVM 寄存器名为 Go 汇编名称。"""
        if reg_name.lower().startswith('f'):
            return reg_name.upper()
        return self.__gnu_to_go_reg_map.get(reg_name.lower(), reg_name.upper())

    def __init__(self):
        logging.info("正在初始化 Translator...")
        self.out = []
        self.mbuf = b''
        self.subr = {}
        self.labels = {}
        self.relocations = {}
        self.aliases = {}
        self.comment_buf = []
        self.sections = {}
        self.data_sizes = {}
        self.pending_lui = {}
        
        # --- 初始化 CFG 属性 ---
        self.labels_to_blocks = {}
        self.blocks = [BasicBlock("__entry__")]
        self.current_block = self.blocks[0]
        self.dead_code = False

        try:
            logging.debug("正在检查 'llvm-mc' 可用性...")
            proc = subprocess.run(['llvm-mc', '--version'], capture_output=True, check=True, text=True)
            logging.info(f"正在使用 {proc.stdout.strip()}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logging.critical(f"严重错误: `llvm-mc` 命令未找到或运行失败: {e}")
            logging.critical("请确保 LLVM 工具链已安装并且 `llvm-mc` 在您的系统路径中。")
            sys.exit(1)
        logging.info("Translator 初始化成功。")

    # ... (省略 _drain, _flush, _lookup, _assemble, _size_*, _emit_* 等未改变的方法) ...
    def _drain(self):
        if self.mbuf:
            val, = struct.unpack('<I', self.mbuf.ljust(4, b'\x00'))
            comment = self.comment_buf.pop(0) if self.comment_buf else f'// {repr(self.mbuf)[1:]}'
            self.out.append(f'    WORD $0x{val:08x}  {comment}')
            self.mbuf = b''

    def _flush(self):
        while len(self.mbuf) >= 8:
            buf, self.mbuf = self.mbuf[:8], self.mbuf[8:]
            low_word, high_word = struct.unpack('<II', buf)
            comment = self.comment_buf.pop(0) if self.comment_buf else ''
            self.out.append(f'    WORD $0x{low_word:08x}; WORD $0x{high_word:08x}  {comment}')
        
        if len(self.mbuf) >= 4:
            buf, self.mbuf = self.mbuf[:4], self.mbuf[4:]
            val, = struct.unpack('<I', buf)
            comment = self.comment_buf.pop(0) if self.comment_buf else ''
            self.out.append(f'    WORD $0x{val:08x}  {comment}')

    def _lookup(self, key: str) -> int:
        clean_key = key.lstrip('.')
        if clean_key not in self.labels:
            raise SyntaxError(f'unresolved reference to {repr(key)}')
        return self.labels[clean_key]

    def _assemble(self, ins: str) -> bytes:
        logging.debug(f"正在汇编指令: '{ins}'")
        obj_file_path, bin_file_path = None, None
        try:
            with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as obj_file:
                obj_file_path = obj_file.name
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as bin_file:
                bin_file_path = bin_file.name
            mc_cmd = ['llvm-mc', '-arch=riscv64', '-filetype=obj', '-mattr=+m,+a,+f,+d,+v', '-o', obj_file_path]
            subprocess.run(mc_cmd, input=ins.encode('utf-8'), capture_output=True, check=True, timeout=5)
            objcopy_cmd = ['llvm-objcopy', '-O', 'binary', '--only-section=.text', obj_file_path, bin_file_path]
            subprocess.run(objcopy_cmd, capture_output=True, check=True, timeout=5)
            with open(bin_file_path, 'rb') as f:
                code = f.read()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr = e.stderr.decode(errors='ignore') if hasattr(e, 'stderr') else "N/A"
            raise RuntimeError(f"llvm tool execution failed for '{ins}'. Stderr:\n{stderr}") from e
        finally:
            if obj_file_path and os.path.exists(obj_file_path): os.remove(obj_file_path)
            if bin_file_path and os.path.exists(bin_file_path): os.remove(bin_file_path)
        
        if not code:
             raise ValueError(f"Assembled instruction resulted in empty binary for '{ins}'")
        return code

    def _size_byte(self, *_) -> int: return 1
    def _size_word(self, *_) -> int: return 2
    def _size_long(self, *_) -> int: return 4
    def _size_quad(self, *_) -> int: return 8
    
    def _size_ascii(self, _: int, cmd: Command) -> int:
        return len(cmd.as_bytes(0)) + (1 if cmd.cmd == '.asciz' else 0)

    def _size_space(self, _: int, cmd: Command) -> int: return cmd.as_int(0)
    def _size_p2align(self, pc: int, cmd: Command) -> int:
        align = 1 << cmd.as_int(0)
        return (align - (pc % align)) % align

    __command_size_tab__ = {
        '.byte': _size_byte, '.short': _size_word,
        '.long': _size_long, '.word': _size_long,
        '.quad': _size_quad, '.ascii': _size_ascii,
        '.asciz': _size_ascii,
        '.space': _size_space, '.p2align': _size_p2align,
        '.zero': _size_space,
    }

    def _emit_raw(self, v: bytes) -> int:
        self.mbuf += v
        self._flush()
        return len(v)

    def _emit_byte(self, _: int, cmd: Command) -> int:
        self.comment_buf.append('// ' + str(cmd))
        return self._emit_raw(struct.pack('<B', Expression(cmd.as_str(0)).eval(self._lookup)))

    def _emit_word(self, _: int, cmd: Command) -> int:
        self.comment_buf.append('// ' + str(cmd))
        return self._emit_raw(struct.pack('<H', Expression(cmd.as_str(0)).eval(self._lookup)))

    def _emit_long(self, _: int, cmd: Command) -> int:
        self.comment_buf.append('// ' + str(cmd))
        return self._emit_raw(struct.pack('<I', Expression(cmd.as_str(0)).eval(self._lookup)))

    def _emit_quad(self, _: int, cmd: Command) -> int:
        self.comment_buf.append('// ' + str(cmd))
        return self._emit_raw(struct.pack('<q', Expression(cmd.as_str(0)).eval(self._lookup)))

    def _emit_ascii(self, _: int, cmd: Command) -> int:
        self.comment_buf.append('// ' + str(cmd))
        data = cmd.as_bytes(0)
        if cmd.cmd == '.asciz':
            data += b'\x00'
        return self._emit_raw(data)

    def _emit_space(self, _: int, cmd: Command) -> int:
        self.comment_buf.append('// ' + str(cmd))
        return self._emit_raw(b'\x00' * cmd.as_int(0))

    def _emit_p2align(self, pc: int, cmd: Command) -> int:
        align = 1 << cmd.as_int(0)
        count = (align - (pc % align)) % align
        if count > 0:
            self.comment_buf.append(f'// {str(cmd)} (padding {count} bytes)')
            return self._emit_raw(b'\x00' * count)
        return 0

    __command_tab__ = {
        '.byte': (_size_byte, _emit_byte), '.short': (_size_word, _emit_word),
        '.long': (_size_long, _emit_long), '.word': (_size_long, _emit_long),
        '.quad': (_size_quad, _emit_quad), 
        '.ascii': (_size_ascii, _emit_ascii), '.asciz': (_size_ascii, _emit_ascii),
        '.space': (_size_space, _emit_space), '.zero': (_size_space, _emit_space),
        '.p2align': (_size_p2align, _emit_p2align),
    }

    def _emit_ref(self, _: int, ref: Label):
        self._drain()
        self.out.append(f'{ref.name}:')

    def _emit_cmd(self, pc: int, cmd: Command, *, dry_run: bool) -> int:
        if cmd.cmd not in self.__command_tab__:
            return 0
        handler = self.__command_tab__[cmd.cmd]
        return self.__command_size_tab__[cmd.cmd](self, pc, cmd) if dry_run else handler[1](self, pc, cmd)

    def _emit_data_command(self, cmd: Command, base_label: str, offset: int):
        cmd_name = cmd.cmd
        
        if cmd_name == '.quad':
            val = Expression(cmd.args[0]).eval(self._lookup)
            low_word = val & 0xFFFFFFFF
            high_word = (val >> 32) & 0xFFFFFFFF
            self.out.append(f'    DATA ·{base_label}+{offset}(SB)/4, ${low_word:#010x}')
            self.out.append(f'    DATA ·{base_label}+{offset+4}(SB)/4, ${high_word:#010x}')
        
        elif cmd_name in ['.long', '.word']:
            val = Expression(cmd.args[0]).eval(self._lookup)
            self.out.append(f'    DATA ·{base_label}+{offset}(SB)/4, ${val:#010x}')

        elif cmd_name == '.short':
            val = Expression(cmd.args[0]).eval(self._lookup)
            self.out.append(f'    DATA ·{base_label}+{offset}(SB)/2, ${val:#06x}')

        elif cmd_name == '.byte':
            val = Expression(cmd.args[0]).eval(self._lookup)
            self.out.append(f'    DATA ·{base_label}+{offset}(SB)/1, ${val:#04x}')

        elif cmd_name in ['.ascii', '.asciz']:
            data = cmd.as_bytes(0)
            if cmd_name == '.asciz':
                data += b'\x00'
            
            i = 0
            while i < len(data):
                remaining = len(data) - i
                
                if remaining >= 8:
                    chunk = data[i:i+8]
                    comment = f'// "{escape_bytes(chunk)}"'
                    val, = struct.unpack('<Q', chunk)
                    self.out.append(f'    DATA ·{base_label}+{offset+i}(SB)/8, ${val:#018x}  {comment}')
                    i += 8
                elif remaining >= 4:
                    chunk = data[i:i+4]
                    comment = f'// "{escape_bytes(chunk)}"'
                    val, = struct.unpack('<I', chunk)
                    self.out.append(f'    DATA ·{base_label}+{offset+i}(SB)/4, ${val:#010x}  {comment}')
                    i += 4
                elif remaining >= 2:
                    chunk = data[i:i+2]
                    comment = f'// "{escape_bytes(chunk)}"'
                    val, = struct.unpack('<H', chunk)
                    self.out.append(f'    DATA ·{base_label}+{offset+i}(SB)/2, ${val:#06x}  {comment}')
                    i += 2
                else:
                    byte = data[i]
                    comment = f'// "{escape_bytes(data[i:i+1])}"'
                    self.out.append(f'    DATA ·{base_label}+{offset+i}(SB)/1, ${byte:#04x}  {comment}')
                    i += 1
        
        elif cmd_name in ['.space', '.zero']:
            size = cmd.as_int(0)
            i = 0
            while i < size:
                remaining = size - i
                chunk_size = 0
                
                if remaining >= 8:
                    chunk_size = 8
                elif remaining >= 4:
                    chunk_size = 4
                elif remaining >= 2:
                    chunk_size = 2
                elif remaining >= 1:
                    chunk_size = 1
                
                if chunk_size > 0:
                    self.out.append(f'    DATA ·{base_label}+{offset+i}(SB)/{chunk_size}, $0')
                    i += chunk_size
                else:
                    break
        
        else:
            logging.warning(f"在 _emit_data_command 中遇到不支持的指令: {cmd}, 跳过。")

    def _process_instruction(self, pc: int, ins: Instruction, *, dry_run: bool) -> int:
        original_src = ins.text.strip()
        if not dry_run:
            logging.debug(f"正在处理指令 at PC={pc}: '{original_src}'")
            if self.mbuf:
                raise RuntimeError(f'unflushed bytes before instruction: {self.mbuf.hex()}')

        lui_match = re.match(r'lui\s+([a-zA-Z0-9]+),\s*%hi\(([_a-zA-Z0-9\.]+)\)', original_src)
        if lui_match:
            reg, symbol = lui_match.groups()
            clean_symbol = symbol.lstrip('.')
            go_symbol = clean_symbol.replace('.', '_')
            
            self.pending_lui[reg] = symbol
            if not dry_run:
                go_reg = self._translate_reg(reg)
                logging.info(f"遇到 LUI: {reg} -> {go_reg} <- hi({symbol}). 生成 MOV 地址指令并等待配对。")
                self.out.append(f'\tMOV $·{go_symbol}(SB), {go_reg}  // {original_src}')
            return 4

        addi_match = re.match(r'addi\s+([a-zA-Z0-9]+),\s*\1,\s*%lo\(([_a-zA-Z0-9\.]+)\)', original_src)
        if addi_match:
            reg, symbol = addi_match.groups()
            if self.pending_lui.get(reg) == symbol:
                del self.pending_lui[reg]
                if not dry_run:
                    logging.info(f"匹配到 ADDI for {reg}, {symbol}. 此指令被覆盖。")
                    self.out.append(f'\t// {original_src} (skipped, covered by MOV address load)')
                return 0

        load_match = re.match(r'(ld|fld|flw)\s+([a-zA-Z0-9]+),\s*%lo\(([_a-zA-Z0-9\.]+)\)\(([a-zA-Z0-9]+)\)', original_src)
        if load_match:
            mnemonic, dst_reg, symbol, base_reg = load_match.groups()
            if self.pending_lui.get(base_reg) == symbol:
                del self.pending_lui[base_reg]
                if not dry_run:
                    go_mnemonic = {'ld': 'MOV', 'fld': 'MOVD', 'flw': 'MOVF'}[mnemonic]
                    go_dst_reg = self._translate_reg(dst_reg)
                    go_base_reg = self._translate_reg(base_reg)
                    logging.info(f"匹配到 {mnemonic.upper()} for {base_reg}, {symbol}. 生成 {go_mnemonic} 加载值指令。")
                    self.out.append(f'\t{go_mnemonic} ({go_base_reg}), {go_dst_reg}  // {original_src}')
                return 4

        call_match = re.match(r'^call\s+([_a-zA-Z0-9\.]+)$', original_src)
        if call_match:
            symbol = call_match.group(1)
            go_symbol = symbol.replace('.', '_')
            if not dry_run:
                logging.info(f"转换 CALL 伪指令 for '{symbol}'")
                self.out.append(f'\tCALL ·{go_symbol}(SB)  // {original_src}')
            return 8 if dry_run else 4 

        branch_match = re.match(r'^(b\w+|j|jal)\s+(.*)$', original_src)
        if branch_match:
            mnemonic, args_str = branch_match.groups()
            args = [arg.strip() for arg in args_str.split(',')]
            label = args[-1]
            
            if label.startswith('.') or self.labels.get(label.lstrip('.')):
                if not dry_run:
                    clean_label = label.lstrip('.').replace('.', '_')
                    go_mnemonic = mnemonic.upper()
                    
                    go_args = [self._translate_reg(arg) for arg in args[:-1]]

                    pseudo_map = {
                        "BEQZ": ("BEQ", "{0}, ZERO"), "BNEZ": ("BNE", "{0}, ZERO"),
                        "BLEZ": ("BGE", "ZERO, {0}"), "BGEZ": ("BGE", "{0}, ZERO"),
                        "BLTZ": ("BLT", "{0}, ZERO"), "BGTZ": ("BLT", "ZERO, {0}"),
                    }

                    if go_mnemonic in pseudo_map and len(go_args) == 1:
                        new_mnemonic, arg_format = pseudo_map[go_mnemonic]
                        go_mnemonic = new_mnemonic
                        go_args_str = arg_format.format(go_args[0]) + ", "
                    else:
                        if go_mnemonic == 'J': go_mnemonic = 'JMP'
                        go_args_str = ", ".join(go_args)
                        if go_args_str: go_args_str += ", "

                    go_asm_line = f'\t{go_mnemonic} {go_args_str}{clean_label}'
                    self.out.append(f'{go_asm_line}  // {original_src}')
                return 4
            
        if not dry_run:
            try:
                encoded_bytes = self._assemble(original_src)
                if len(encoded_bytes) % 4 != 0:
                    logging.warning(f"汇编结果大小不是4的倍数: {len(encoded_bytes)} bytes for '{original_src}'")
                
                i = 0
                while i < len(encoded_bytes):
                    chunk = encoded_bytes[i:i+4]
                    if len(chunk) < 4:
                        chunk = chunk.ljust(4, b'\x00')
                    
                    val, = struct.unpack('<I', chunk)
                    comment = f'// {original_src}' if i == 0 else ''
                    self.out.append(f'\tWORD $0x{val:08x}  {comment}')
                    i += 4
                return len(encoded_bytes)

            except (RuntimeError, ValueError) as e:
                logging.error(f"汇编指令失败 '{original_src}': {e}")
                self.out.append(f'\t// FAILED TO ASSEMBLE: {original_src}')
                return 4
        
        return 4

    def _scan_and_parse(self, src: str) -> dict[str, list]:
        logging.info("开始扫描与解析源码...")
        sections = {'.text': []}
        current_section = '.text'
        lines = src.splitlines()
        ignored_commands = {'.file', '.attribute', '.ident', '.addrsig', '.type', '.size', '.globl', '.data'}

        for line in lines:
            cleaned_line = Line.remove_comments(line)
            if not cleaned_line: continue

            if cleaned_line.startswith('.section'):
                match = re.match(r'\.section\s+([.\w]+)', cleaned_line)
                if match:
                    current_section = match.group(1).replace('.', '_')
                    if current_section not in sections:
                        sections[current_section] = []
                    logging.info(f"切换到 section: {current_section}")
                continue

            if cleaned_line.strip() == '.text':
                current_section = '.text'
                logging.info(f"切换到 section: {current_section}")
                continue

            if cleaned_line.endswith(':'):
                label_name = cleaned_line[:-1]
                if label_name.startswith('.Lfunc_end'): continue
                sections[current_section].append(Label(label_name.lstrip('.')))
            elif cleaned_line.startswith('.'):
                cmd = Command.parse(cleaned_line)
                if cmd.cmd in ignored_commands: continue
                if cmd.cmd == '.p2align': continue
                sections[current_section].append(cmd)
            else:
                sections[current_section].append(Instruction.parse(cleaned_line))
        
        logging.info(f"扫描与解析完成. 共解析 {len(sections)} 个 section。")
        return sections

    # --- 新增: CFG 构建方法 ---
    def _build_cfg(self, text_items: list):
        logging.info("开始构建控制流图 (CFG)...")
        
        # Helper to get or create a block for a label
        def get_or_create_block(name: str) -> BasicBlock:
            if name not in self.labels_to_blocks:
                self.labels_to_blocks[name] = BasicBlock(name)
                self.blocks.append(self.labels_to_blocks[name])
            return self.labels_to_blocks[name]

        for item in text_items:
            if isinstance(item, Label):
                label_name = item.name
                target_block = get_or_create_block(label_name)
                target_block.body.append(item)

                if not self.dead_code:
                    self.current_block.link_to(target_block)
                
                self.current_block = target_block
                self.dead_code = False
            
            elif isinstance(item, Instruction):
                if self.dead_code:
                    logging.debug(f"跳过死代码指令: {item.text}")
                    continue

                self.current_block.body.append(item)

                if item.is_terminator:
                    target_label = item.get_branch_target()
                    if target_label:
                        target_block = get_or_create_block(target_label)
                        self.current_block.jump_to(target_block)

                    if item.is_conditional_branch:
                        # Fall-through block
                        fallthrough_block = BasicBlock(f"__fallthrough_{len(self.blocks)}")
                        self.blocks.append(fallthrough_block)
                        self.current_block.link_to(fallthrough_block)
                        self.current_block = fallthrough_block
                    else: # Unconditional branch or return
                        self.dead_code = True

            elif isinstance(item, Command):
                 if not self.dead_code:
                    self.current_block.body.append(item)
        
        logging.info(f"CFG 构建完成. 共 {len(self.blocks)} 个基本块。")

    # --- 新增: 栈分析核心方法 ---
    def _trace_instructions(self, bb: BasicBlock) -> int:
        """计算单个基本块内的最大栈深度"""
        max_sp_in_block = 0
        current_sp = 0
        for item in bb.body:
            if isinstance(item, Instruction):
                adj = item.stack_adjustment
                if adj is not None:
                    # addi sp, sp, -N 表示分配栈，所以栈深度增加
                    current_sp -= adj
                    if current_sp > max_sp_in_block:
                        max_sp_in_block = current_sp
        return max_sp_in_block

    def _trace_block(self, bb: BasicBlock) -> int:
        """递归遍历CFG计算从该块开始的最大栈深度"""
        if bb.max_sp >= 0: # 已计算
            return bb.max_sp
        if bb.max_sp == -2: # 正在计算，遇到环
            logging.warning(f"在栈分析中检测到环，涉及块 {bb.name}，假定环内栈大小不变。")
            return 0

        bb.max_sp = -2 # 标记为正在计算

        sp_in_this_block = self._trace_instructions(bb)
        
        sp_from_jump = 0
        if bb.jump:
            sp_from_jump = self._trace_block(bb.jump)

        sp_from_next = 0
        if bb.next:
            sp_from_next = self._trace_block(bb.next)
        
        # 最大栈深度 = 当前块的深度 + 后续路径中最大的深度
        bb.max_sp = sp_in_this_block + max(sp_from_jump, sp_from_next)
        logging.debug(f"块 {bb.name} 的栈分析完成: "
                      f"块内深度={sp_in_this_block}, "
                      f"跳转路径深度={sp_from_jump}, "
                      f"顺序路径深度={sp_from_next}, "
                      f"总计={bb.max_sp}")
        return bb.max_sp

    def stack_size(self, func_name: str) -> int:
        """计算指定函数的最大栈大小"""
        clean_name = func_name.replace('.', '_')
        if clean_name not in self.labels_to_blocks:
            logging.error(f"无法找到函数 '{func_name}' 的入口基本块。")
            return 0
        
        start_block = self.labels_to_blocks[clean_name]
        logging.info(f"--- 开始对函数 '{func_name}' 进行栈分析 ---")
        size = self._trace_block(start_block)
        logging.info(f"--- 函数 '{func_name}' 的栈分析完成，计算得最大栈大小: {size} 字节 ---")
        return size

    def translate_c_to_binary(self, src: str, proto: Prototypes, *, name: str):
        logging.info(f"--- 开始将 C 汇编翻译为二进制块 '{name}' ---")
        self.sections = self._scan_and_parse(src)
        
        text_ins = self.sections.get('.text', [])
        
        # --- 阶段 1: 构建 CFG ---
        self._build_cfg(text_ins)

        # --- 阶段 2: 计算所有 section 的标签地址和重定位 ---
        logging.info("Pass 1: 计算所有标签的地址和重定位...")
        pc = 0
        self.pending_lui.clear()
        for v in text_ins:
            if isinstance(v, Label):
                if v.name in self.labels: raise SyntaxError(f"Duplicate label: {v}")
                self.labels[v.name] = pc
                logging.debug(f"Pass 1: 文本标签 '{v}' -> '{v.name}' 位于 PC={pc} (0x{pc:x})")
            elif isinstance(v, Command):
                pc += self._emit_cmd(pc, v, dry_run=True)
            elif isinstance(v, Instruction):
                pc = (pc + 3) & ~3
                pc += self._process_instruction(pc, v, dry_run=True)
        
        data_pc = 0
        for sec_name, sec_ins in self.sections.items():
            if sec_name == '.text': continue
            logging.info(f"Pass 1: 计算 {sec_name} section 的标签地址...")
            current_label = None
            label_start_pc = 0
            for v in sec_ins:
                if isinstance(v, Label):
                    if current_label:
                        self.data_sizes[current_label] = data_pc - label_start_pc
                    align = 8
                    data_pc = (data_pc + align - 1) & ~(align - 1)
                    if v.name in self.labels: raise SyntaxError(f"Duplicate label: {v}")
                    self.labels[v.name] = data_pc
                    current_label = v.name
                    label_start_pc = data_pc
                    logging.debug(f"Pass 1: 数据标签 '{v.name}' 位于 offset={data_pc} (0x{data_pc:x})")
                elif isinstance(v, Command):
                    size_func = self.__command_size_tab__.get(v.cmd, lambda *_: 0)
                    data_pc += size_func(self, data_pc, v)
            if current_label:
                self.data_sizes[current_label] = data_pc - label_start_pc
        logging.info(f"Pass 1 完成.")
        
        # --- 阶段 3: 生成 TEXT 和 DATA 块 (代码生成) ---
        logging.info("Pass 2: 生成 .text section 的 TEXT 块...")
        self.out.append(f'TEXT ·{name}(SB), NOSPLIT, $0')
        self.out.append('    NO_LOCAL_POINTERS')
        current_pc = 0
        self.pending_lui.clear()
        for v in text_ins:
            if isinstance(v, Label):
                aligned_pc = (current_pc + 3) & ~3
                padding = aligned_pc - current_pc
                if padding > 0: self._emit_raw(b'\x00' * padding)
                current_pc = aligned_pc
                self._emit_ref(current_pc, v)
            elif isinstance(v, Command):
                current_pc += self._emit_cmd(current_pc, v, dry_run=False)
            elif isinstance(v, Instruction):
                aligned_pc = (current_pc + 3) & ~3
                padding = aligned_pc - current_pc
                if padding > 0: self._emit_raw(b'\x00' * padding)
                current_pc = aligned_pc
                size = self._process_instruction(current_pc, v, dry_run=False)
                current_pc += size
        self._drain()
        logging.info("Pass 2 完成. TEXT 块已生成。")

        for sec_name, sec_ins in self.sections.items():
            if sec_name == '.text' or not sec_ins: continue
            logging.info(f"Pass 2.5: 生成 {sec_name} section 的 DATA 块...")
            current_base_label = None
            current_offset = 0
            for v in sec_ins:
                if isinstance(v, Label):
                    current_base_label = v.name
                    current_offset = self.labels[current_base_label]
                    self.out.append('')
                    size = self.data_sizes.get(current_base_label, 0)
                    self.out.append(f'GLOBL ·{current_base_label}(SB), NOPTR, ${size}')
                elif isinstance(v, Command):
                    if not current_base_label:
                        logging.error(f"发现没有前导标签的数据指令: {v}, 跳过。")
                        continue
                    if v.cmd == '.p2align':
                        align = 1 << v.as_int(0)
                        aligned_offset = (current_offset + align - 1) & ~(align - 1)
                        current_offset = aligned_offset
                        continue
                    relative_offset = current_offset - self.labels[current_base_label]
                    self._emit_data_command(v, current_base_label, relative_offset)
                    size_func = self.__command_size_tab__.get(v.cmd, lambda *_: 0)
                    current_offset += size_func(self, current_offset, v)
            self._drain()

        # --- 阶段 4: 填充 self.subr, 并使用栈分析结果 ---
        for go_func_name, p in sorted(proto.items()):
            c_label_to_find = p.c_name()
            subr_key_name = go_func_name
            if go_func_name.startswith('__'):
                c_label_to_find = go_func_name[2:]
                subr_key_name = go_func_name[1:]
                logging.info(f"应用名称匹配规则: Go 原型 '{go_func_name}' -> C 标签 '{c_label_to_find}', Subr 键名 '{subr_key_name}'")
            
            c_func_label = next((lbl for lbl in self.labels if lbl == c_label_to_find), None)
            if c_func_label:
                # --- 关键修改: 调用栈分析 ---
                stack_size = self.stack_size(c_label_to_find)
                self.subr[subr_key_name] = Subroutine(offset=self.labels[c_func_label], stack_size=stack_size)
                logging.info(f"匹配成功: '{go_func_name}' 已链接到 C 函数 '{c_label_to_find}' (Subr: '{subr_key_name}')")
            else:
                logging.warning(f"未在 C 汇编中找到 Go 原型 '{go_func_name}' 的匹配标签 (尝试查找: '{c_label_to_find}')")

        logging.info("--- C 汇编翻译完成 ---")

    def generate_go_wrappers(self, proto: Prototypes, c_code_base_name: str):
        logging.info(f"--- 开始生成 Go 汇编 Wrapper (C 代码基地址: ·{c_code_base_name}) ---")
        int_regs = [f'X{10 + i}' for i in range(8)]
        float_regs = [f'F{10 + i}' for i in range(8)]
        for name, p in proto.items():
            # --- 修改: 使用正确的 subr 键名 ---
            subr_key_name = name[1:] if name.startswith('__') else name
            if subr_key_name not in self.subr: continue
            subr_info = self.subr[subr_key_name]
            
            logging.info(f"正在为函数 ·{name} 生成 Wrapper")
            self.out.append('')
            frame_size = sum(arg.size for arg in p.args) + (p.retv.size if p.retv else 0)
            self.out.append(f'TEXT ·{name}(SB), $0-{frame_size}')
            self.out.append('    NO_LOCAL_POINTERS')
            self.out.append('')
            self.out.append(f'_entry_{name}:')
            self.out.append('    MOV     16(g), X31')
            # --- 修改: 使用计算出的栈大小 ---
            # Go 运行时需要知道 C 函数的栈帧大小，以便进行栈检查
            # 我们需要为 C 函数的栈帧 + Go wrapper 保存 RA/FP 等所需的额外空间留出余量
            # 这里保守地增加 80 字节
            required_stack = subr_info.stack_size + 80
            self.out.append(f'    ADD     $-{required_stack}, SP, X30')
            self.out.append(f'    BLTU    X30, X31, _stack_grow_{name}')
            self.out.append('')
            self.out.append(f'_{name}:')
            current_offset, int_reg_idx, float_reg_idx = 0, 0, 0
            for arg in p.args:
                if arg.is_float:
                    if float_reg_idx < len(float_regs):
                        self.out.append(f'    {"MOVD" if arg.size == 8 else "MOVW"}   {arg.name}+{current_offset}(FP), {float_regs[float_reg_idx]}')
                        float_reg_idx += 1
                else:
                    if int_reg_idx < len(int_regs):
                        self.out.append(f'    MOV     {arg.name}+{current_offset}(FP), {int_regs[int_reg_idx]}')
                        int_reg_idx += 1
                current_offset += arg.size
            self.out.append(f'    CALL    ·{c_code_base_name}+{subr_info.offset}(SB)')
            if p.retv:
                if p.retv.is_float:
                    self.out.append(f'    {"MOVD" if p.retv.size == 8 else "MOVW"}   F10, {p.retv.name}+{current_offset}(FP)')
                else:
                    self.out.append(f'    MOV     X10, {p.retv.name}+{current_offset}(FP)')
            self.out.append('    RET')
            self.out.append('')
            self.out.append(f'_stack_grow_{name}:')
            self.out.append('    MOV     X1, X3')
            self.out.append('    CALL    runtime·morestack_noctxt(SB)')
            self.out.append(f'    JMP     _entry_{name}')
        logging.info("--- Go 汇编 Wrapper 生成完成 ---")

# ... (main 函数和 setup_logging 保持不变) ...
def setup_logging(output_s_file: str):
    """根据输出文件名配置日志记录。"""
    log_dir = "asm2asm_log"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        sys.exit(f"严重错误: 无法创建日志目录 '{log_dir}': {e}")

    log_base_name = os.path.splitext(os.path.basename(output_s_file))[0]
    log_file_path = os.path.join(log_dir, f"{log_base_name}.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        filename=log_file_path,
        filemode='w',
        force=True
    )
    logging.info(f"日志将输出到: {log_file_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert RISC-V LLVM asm to Go asm.')
    parser.add_argument('proto_file', type=str, help='The Go file that declares Go function prototypes.')
    parser.add_argument('asm_files', type=str, nargs='+', help='One or more LLVM assembly files.')
    
    args = parser.parse_args()

    proto_file_path = args.proto_file
    src_files = args.asm_files
    
    proto_base_name = os.path.splitext(proto_file_path)[0]

    # 打印proto_base_name
    logging.info(f"原型文件基础名: {proto_base_name}")

    out_file = proto_base_name + '.s'

    setup_logging(out_file)

    if not os.path.exists(proto_file_path):
        sys.exit(f"严重错误: 原型文件 '{proto_file_path}' 未找到。")

    pkg, proto = "main", Prototypes()
    try:
        with open(proto_file_path, 'r') as fp:
            pkg, proto = Prototypes.parse(CallingConvention.riscv64(), fp.read())
    except SyntaxError as e:
        sys.exit(f'严重错误: 原型文件 "{proto_file_path}" 语法错误: {e}')
    
    src = []
    for fn in src_files:
        with open(fn, 'r') as fp:
            src.extend(fp.read().splitlines())
    
    asm = Translator()
    asm.out.extend([
        '// +build !noasm !appengine',
        '// Code generated by asm2asm, DO NOT EDIT.',
        f'// filename: {os.path.basename(out_file)}',
        '',
        '#include "go_asm.h"',
        '#include "funcdata.h"',
        '#include "textflag.h"',
        '',
    ])
    
    c_code_block_name = f"__{os.path.splitext(os.path.basename(proto_file_path))[0]}_entry__"
    asm.translate_c_to_binary(src='\n'.join(src), proto=proto, name=c_code_block_name)
    
    if proto.items():
        asm.generate_go_wrappers(proto=proto, c_code_base_name=c_code_block_name)
    
    with open(out_file, 'w') as fp:
        fp.write('\n'.join(asm.out) + '\n')
    
    subr_file_name = os.path.join(os.path.dirname(out_file), make_subr_filename(out_file))
    save_subr_refs(pkg=pkg, subr=asm.subr, base=c_code_block_name, name=subr_file_name)
    
    logging.info(f"成功生成 '{out_file}' 和 '{subr_file_name}'。")

if __name__ == '__main__':
    main()