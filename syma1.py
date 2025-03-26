# -*- coding: utf-8 -*-
import math
import numpy as np
import io
import sys
from collections import deque
import readline # Optional: for REPL history and editing
import re
import platform # Для определения ОС (пока не используется для getch)

# --- Вспомогательные функции (is_float, is_int - без изменений) ---
# ... (is_float, is_int) ...
def is_float(s):
    if isinstance(s, (int, float, complex)): return True
    if not isinstance(s, str) or not s or s == '-' or s == '.': return False
    if s.lower() in ['inf', '-inf', 'nan']: return True
    try: float(s); return True
    except ValueError: return False
def is_int(s):
    if isinstance(s, int): return True
    if isinstance(s, float): return s.is_integer()
    if not isinstance(s, str) or not s or s == '-': return False
    try: f=float(s); return not math.isinf(f) and not math.isnan(f) and f.is_integer()
    except ValueError: return False


# --- Класс Интерпретатора SyMaInterpreter ---
class SyMaInterpreter:
    def __init__(self):
        self.stack = deque()
        self.variables = {
            'π': math.pi, '𝑒': math.e, '𝑖': 1j, '∞': float('inf'), '⊤': True, '⊥': False,
            'pi': math.pi, 'tau': 2 * math.pi, 'e': math.e, 'j': 1j,
            'inf': float('inf'), 'True': True, 'False': False,
        }
        self.operations = {}
        self.stream = None
        self._register_core_ops()

    # ... (push, pop, peek - без изменений) ...
    def push(self, value): self.stack.append(value)
    def pop(self):
        if not self.stack: raise IndexError("Stack underflow: Attempt to pop from empty stack")
        return self.stack.pop()
    def peek(self, index=-1):
        n = len(self.stack)
        if n == 0 or index < -n or index >= n: raise IndexError(f"Stack peek error: Invalid index {index} for stack size {n}")
        return self.stack[index]


    def _register_core_ops(self):
        symbol_ops = {
            # ... (предыдущие символы) ...
            '┐': self.op_drop, '┤': self.op_dup, '⇌': self.op_swap, '┴': self.op_over,
            '↻': self.op_rot, '↺': self.op_rot_neg, '☰': self.op_depth, '¡': self.op_clear_stack,
            '+': self.op_add, '-': self.op_sub, '×': self.op_mul, '÷': self.op_div, '^': self.op_pow,
            '√': self.op_sqrt, '∣': self.op_abs_det_norm, '%': self.op_mod, '‼': self.op_factorial,
            '<': self.op_lt, '>': self.op_gt, '=': self.op_eq, '≠': self.op_neq, '≤': self.op_le, '≥': self.op_ge,
            '¬': self.op_not, '∧': self.op_and, '∨': self.op_or, '⊕': self.op_xor,
            '∿': self.op_sin, 'κ': self.op_cos, 'τ': self.op_tan, '㏑': self.op_ln, '㏒': self.op_log10,
            'ε': self.op_exp, '⌈': self.op_ceil, '⌊': self.op_floor, '◎': self.op_round,
            '→': self.op_store, '!': self.op_execute,
            '▼': self.op_print_ln, '▽': self.op_print, '✎': self.op_print_raw_str, '▲': self.op_input,
            '@': self.op_matmul, '⋅': self.op_dot, '⍉': self.op_transpose, '\\': self.op_solve,
            '𝟙': self.op_identity, '📐': self.op_shape, '↕': self.op_range,
            '?': self.op_conditional, 'λ': self.op_while, '⌢': self.op_concat,
            # *** НОВЫЕ ОПЕРАЦИИ ТЕРМИНАЛА ***
            '⚑': self.op_gotoxy, # Символ для gotoxy (флажок)
            '□': self.op_cls,    # Символ для cls (очистить экран)
         }
        self.operations.update(symbol_ops)
        aliases = {
            # ... (предыдущие алиасы) ...
            'drop': '┐', 'dup': '┤', 'swap': '⇌', 'over': '┴', 'rot': '↻', 'nrot': '↺',
            'depth': '☰', 'clear': '¡', 'add': '+', 'sub': '-', 'mul': '×', 'div': '÷',
            'pow': '^', 'sqrt': '√', 'abs': '∣', 'det': '∣', 'norm': '∣', 'mod': '%', 'fact': '‼',
            'lt': '<', 'gt': '>', 'eq': '=', 'neq': '≠', 'le': '≤', 'ge': '≥',
            'not': '¬', 'and': '∧', 'or': '∨', 'xor': '⊕', 'sin': '∿', 'cos': 'κ', 'tan': 'τ',
            'ln': '㏑', 'log10': '㏒', 'exp': 'ε', 'ceil': '⌈', 'floor': '⌊', 'round': '◎',
            'store': '→', 'assign': '→', 'exec': '!', 'execute': '!',
            'println': '▼', 'print': '▽', 'prints': '✎', 'input': '▲', 'matmul': '@',
            'dot': '⋅', 'transpose': '⍉', 'solve': '\\', 'identity': '𝟙', 'eye': '𝟙',
            'shape': '📐', 'range': '↕', 'if': '?', 'while': 'λ', 'concat': '⌢',
             # *** НОВЫЕ АЛИАСЫ ***
            'gotoxy': '⚑',
            'cls': '□', 'clear_screen': '□',
        }
        for alias, symbol in aliases.items():
            if symbol in self.operations: self.operations[alias] = self.operations[symbol]

    def _check_stack_depth(self, required, op_symbol):
        if len(self.stack) < required:
            raise IndexError(f"Stack underflow for '{op_symbol}' (need {required} operands, have {len(self.stack)})")

    # --- Операции (Добавлены op_gotoxy, op_cls) ---
    # ... (остальные op_* без изменений) ...
    def op_drop(self): self.pop()
    def op_dup(self): self._check_stack_depth(1, '┤'); self.push(self.peek())
    def op_swap(self): self._check_stack_depth(2, '⇌'); b,a=self.pop(),self.pop(); self.push(b); self.push(a)
    def op_over(self): self._check_stack_depth(2, '┴'); b=self.pop(); a=self.peek(-1); self.push(b); self.push(a)
    def op_rot(self): self._check_stack_depth(3, '↻'); c,b,a=self.pop(),self.pop(),self.pop(); self.push(b); self.push(c); self.push(a)
    def op_rot_neg(self): self._check_stack_depth(3, '↺'); c,b,a=self.pop(),self.pop(),self.pop(); self.push(c); self.push(a); self.push(b)
    def op_depth(self): self.push(len(self.stack))
    def op_clear_stack(self): self.stack.clear()
    def op_add(self): self._check_stack_depth(2, '+'); b, a = self.pop(), self.pop(); self.push(str(a) + str(b) if isinstance(a, str) or isinstance(b, str) else np.add(a, b))
    def op_sub(self): self._check_stack_depth(2, '-'); b, a = self.pop(), self.pop(); self.push(np.subtract(a, b))
    def op_mul(self): self._check_stack_depth(2, '×'); b, a = self.pop(), self.pop(); self.push(np.multiply(a, b))
    def op_div(self):
        self._check_stack_depth(2, '÷'); b, a = self.pop(), self.pop()
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.divide(a, b)
            if isinstance(r, np.ndarray): r[np.isinf(r)]=np.inf; r[np.isnan(r)]=np.nan
            elif np.isinf(r): r=np.inf
            elif np.isnan(r): r=np.nan
        self.push(r)
    def op_pow(self): self._check_stack_depth(2, '^'); b, a = self.pop(), self.pop(); self.push(np.power(a, b))
    def op_mod(self): self._check_stack_depth(2, '%'); b, a = self.pop(), self.pop(); self.push(np.mod(a, b))
    def op_factorial(self):
        a = self.pop()
        try:
            if isinstance(a, np.ndarray): raise TypeError("Factorial undefined for arrays")
            v = int(a)
            if v != a or v < 0: raise ValueError("Factorial only defined for non-negative integers")
            result = math.factorial(v); self.push(result)
        except (ValueError, TypeError) as e: self.push(a); raise ValueError(f"Factorial error '‼': {e}") from e
        except OverflowError as e: self.push(a); raise OverflowError(f"Factorial error '‼': Result too large - {e}") from e
    def op_abs_det_norm(self):
        a = self.pop()
        if isinstance(a, (int, float, complex)): self.push(abs(a))
        elif isinstance(a, np.ndarray):
            if a.ndim == 0: self.push(abs(a.item()))
            elif a.ndim == 1: self.push(np.linalg.norm(a))
            elif a.ndim == 2 and a.shape[0] == a.shape[1]:
                try: det_val = np.linalg.det(a); self.push(det_val)
                except np.linalg.LinAlgError as e: self.push(a); raise ValueError(f"Determinant error: {e}") from e
            else: self.push(np.linalg.norm(a))
        else: self.push(a); raise TypeError(f"Unsupported type for '∣': {type(a)}")
    def op_lt(self): self._check_stack_depth(2, '<'); b,a=self.pop(),self.pop(); self.push(a < b)
    def op_gt(self): self._check_stack_depth(2, '>'); b,a=self.pop(),self.pop(); self.push(a > b)
    def op_eq(self):
        self._check_stack_depth(2, '='); b,a=self.pop(),self.pop()
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b): self.push(True)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            try: are_equal = np.array_equal(a, b, equal_nan=True); self.push(are_equal)
            except TypeError: self.push(False)
        else: self.push(a == b)
    def op_neq(self):
        self._check_stack_depth(2, '≠'); b,a=self.pop(),self.pop()
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b): self.push(False)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
             try: are_equal = np.array_equal(a, b, equal_nan=True); self.push(not are_equal)
             except TypeError: self.push(True)
        else: self.push(a != b)
    def op_le(self): self._check_stack_depth(2, '≤'); b,a=self.pop(),self.pop(); self.push(a <= b)
    def op_ge(self): self._check_stack_depth(2, '≥'); b,a=self.pop(),self.pop(); self.push(a >= b)
    def op_not(self): self.push(not self.pop())
    def op_and(self): self._check_stack_depth(2, '∧'); b,a=self.pop(),self.pop(); self.push(a and b)
    def op_or(self): self._check_stack_depth(2, '∨'); b,a=self.pop(),self.pop(); self.push(a or b)
    def op_xor(self): self._check_stack_depth(2, '⊕'); b,a=self.pop(),self.pop(); self.push(bool(a) ^ bool(b))
    def op_sin(self): self.push(np.sin(self.pop()))
    def op_cos(self): self.push(np.cos(self.pop()))
    def op_tan(self): self.push(np.tan(self.pop()))
    def op_ln(self): self.push(np.lib.scimath.log(self.pop()))
    def op_log10(self): self.push(np.lib.scimath.log10(self.pop()))
    def op_exp(self): self.push(np.exp(self.pop()))
    def op_ceil(self): self.push(np.ceil(self.pop()))
    def op_floor(self): self.push(np.floor(self.pop()))
    def op_round(self): self.push(np.round(self.pop()))
    def op_sqrt(self): self.push(np.lib.scimath.sqrt(self.pop()))
    def op_store(self):
        self._check_stack_depth(2, '→/=/store'); name, val = self.pop(), self.pop()
        if not isinstance(name, str): self.push(val); self.push(name); raise TypeError(f"Variable name must be string, got {type(name)}")
        if not name or not (name[0].isalpha() or name[0] == '_'): raise ValueError(f"Invalid variable name: '{name}' (must start with letter or _)")
        if not all(c.isalnum() or c == '_' for c in name): raise ValueError(f"Invalid variable name: '{name}' (allowed: letters, numbers, _)")
        self.variables[name] = val
    def _execute_block(self, code_block_str):
        is_debug_on = getattr(self, 'DEBUG_MODE', False)
        if is_debug_on: print(f"DEBUG >> Enter execute_block: '{code_block_str[:30]}...'")
        if not isinstance(code_block_str, str) or not code_block_str.startswith('{') or not code_block_str.endswith('}'): raise TypeError(f"Execute target must be code block '{{...}}', got: {code_block_str}")
        temp_interpreter = SyMaInterpreter(); temp_interpreter.stack = self.stack; temp_interpreter.variables = self.variables; temp_interpreter.operations = self.operations
        setattr(temp_interpreter, 'DEBUG_MODE', is_debug_on)
        inner_code = code_block_str[1:-1]
        try:
            temp_interpreter.run(inner_code)
            if is_debug_on: print(f"DEBUG << Exit execute_block: '{code_block_str[:30]}...'")
        except Exception as e:
             if is_debug_on: print(f"DEBUG !! Error Exit execute_block: '{code_block_str[:30]}...'")
             e.args = (f"Error within block '{code_block_str[:20]}...': {e}",) + e.args[1:]
             raise
    def op_execute(self):
        code_block = self.pop()
        if not isinstance(code_block, str) or not code_block.startswith('{') or not code_block.endswith('}'): self.push(code_block); raise TypeError(f"Execute target '!' must be string '{{...}}', got: {type(code_block)}")
        self._execute_block(code_block)
    def op_print_ln(self): print(self.pop() if self.stack else "[Stack empty]")
    def op_print(self): print(self.pop() if self.stack else "[Stack empty]", end='')
    def op_print_raw_str(self):
        s = self.pop()
        if not isinstance(s, str): self.push(s); raise TypeError(f"Raw print '✎' target not string, got {type(s)}")
        print(s, end='')
    def op_input(self):
        line = None
        try: line = input()
        except EOFError: raise EOFError("EOF encountered during input '▲'")
        except Exception as e: raise RuntimeError(f"Unexpected error during input '▲': {e}") from e
        push = self.push
        if is_int(line): push(int(float(line)))
        elif is_float(line): push(float(line))
        elif line.lower() in ['true', '⊤']: push(True)
        elif line.lower() in ['false', '⊥']: push(False)
        else: push(line)
    def op_matmul(self): self._check_stack_depth(2, '@'); B, A = self.pop(), self.pop(); self.push(np.matmul(A, B))
    def op_dot(self): self._check_stack_depth(2, '⋅'); B, A = self.pop(), self.pop(); self.push(np.dot(A, B))
    def op_transpose(self): self.push(np.transpose(self.pop()))
    def op_solve(self):
        self._check_stack_depth(2, '\\'); b, A = self.pop(), self.pop()
        try:
             if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]: raise ValueError(f"Matrix A must be square 2D, got {getattr(A, 'shape', 'N/A')}")
             if not isinstance(b, np.ndarray) or b.ndim == 0 or b.ndim > 2 : raise ValueError(f"Vector b must be 1D/2D, got {getattr(b, 'shape', 'N/A')}")
             if A.shape[0] != b.shape[0]: raise ValueError(f"Incompatible shapes A ({A.shape}), b ({b.shape})")
             self.push(np.linalg.solve(A, b))
        except np.linalg.LinAlgError as e: self.push(A); self.push(b); raise ValueError(f"Linear solve error: {e}") from e
        except ValueError as e: self.push(A); self.push(b); raise e
    def op_identity(self):
        n_val = self.pop()
        try:
            n = int(n_val)
            if n < 0: raise ValueError("Dimension must be non-negative")
            self.push(np.eye(n))
        except (ValueError, TypeError) as e: self.push(n_val); raise TypeError(f"Identity '𝟙' requires a non-negative integer, got: {n_val} ({type(n_val).__name__}) - {e}") from e
    def op_shape(self): a = self.pop(); self.push(np.array(getattr(a, 'shape', [])))
    def op_range(self):
        stop = self.pop();
        try:
            stop_val = float(stop)
            self.push(np.arange(stop_val))
        except (ValueError, TypeError) as e: self.push(stop); raise TypeError(f"Range '↕' requires a number, got: {stop} ({type(stop).__name__})") from e
    def op_conditional(self):
        if len(self.stack) < 2: raise IndexError("Stack underflow for '?'")
        arg1, arg2 = self.pop(), self.pop()
        is_arg1_blk = isinstance(arg1, str) and arg1.startswith('{') and arg1.endswith('}')
        is_arg2_blk = isinstance(arg2, str) and arg2.startswith('{') and arg2.endswith('}')
        if is_arg1_blk and is_arg2_blk: fb, tb = arg1, arg2; self._check_stack_depth(1, '? (condition)'); cond = self.pop()
        elif is_arg1_blk: fb, tb, cond = None, arg1, arg2
        else: self.push(arg2); self.push(arg1); raise TypeError("Invalid arguments for '?' (expected code block before '?')")
        if not isinstance(cond, (bool, np.bool_)):
             self.push(cond); self.push(tb);
             if fb: self.push(fb)
             raise TypeError(f"Condition '?' must be boolean, got {type(cond)}")
        if cond: self._execute_block(tb)
        elif fb is not None: self._execute_block(fb)
    def op_while(self):
        self._check_stack_depth(2, 'λ')
        body_b = self.pop(); cond_b = self.pop()
        if not (isinstance(body_b,str) and body_b.startswith('{') and body_b.endswith('}')) or \
           not (isinstance(cond_b,str) and cond_b.startswith('{') and cond_b.endswith('}')):
            self.push(cond_b); self.push(body_b); raise TypeError("While 'λ' requires code blocks '{...}'")
        is_debug_on = getattr(self, 'DEBUG_MODE', False)
        if is_debug_on: print(f"DEBUG Enter WHILE: Cond='{cond_b[:20]}...', Body='{body_b[:20]}...'")
        loop_interpreter = SyMaInterpreter(); loop_interpreter.stack = self.stack; loop_interpreter.variables = self.variables; loop_interpreter.operations = self.operations
        setattr(loop_interpreter, 'DEBUG_MODE', is_debug_on)
        cond_c, body_c = cond_b[1:-1], body_b[1:-1]; max_iter, i = 10000, 0
        while i < max_iter:
            if is_debug_on: print(f"DEBUG WHILE iter {i}: Checking condition...")
            try: loop_interpreter.run(cond_c)
            except Exception as e:
                 if is_debug_on: print(f"DEBUG WHILE iter {i}: Error in condition")
                 raise Exception(f"Error in 'λ' condition: {e}") from e
            self._check_stack_depth(1, 'λ (condition result)'); cond_r = self.pop()
            if is_debug_on: print(f"DEBUG WHILE iter {i}: Condition result = {cond_r}")
            if not isinstance(cond_r,(bool, np.bool_)): self.push(cond_r); self.push(cond_b); self.push(body_b); raise TypeError(f"'λ' condition result not boolean, got {type(cond_r)}")
            if not cond_r:
                if is_debug_on: print(f"DEBUG WHILE iter {i}: Condition False, exiting loop.")
                break
            if is_debug_on: print(f"DEBUG WHILE iter {i}: Executing body...")
            try: loop_interpreter.run(body_c)
            except Exception as e:
                 if is_debug_on: print(f"DEBUG WHILE iter {i}: Error in body")
                 raise Exception(f"Error in 'λ' body: {e}") from e
            if is_debug_on: print(f"DEBUG WHILE iter {i}: Body executed.")
            i += 1
            if i == max_iter: print(f"[Warning] Max iterations ({max_iter}) reached for 'λ'", file=sys.stderr); break
        if is_debug_on: print(f"DEBUG Exit WHILE loop.")
    def op_concat(self): self._check_stack_depth(2, '⌢'); b,a = self.pop(),self.pop(); self.push(str(a) + str(b))

    # *** НОВЫЕ ОПЕРАЦИИ ***
    def op_gotoxy(self):
        """Moves cursor to (X, Y). Stack: Y X ->"""
        self._check_stack_depth(2, '⚑/gotoxy')
        x = self.pop() # Column (from right)
        y = self.pop() # Row (from right)
        try:
            row = int(y) + 1 # ANSI is 1-based
            col = int(x) + 1
            if row < 1 or col < 1: return # Ignore out of bounds? Or raise error?
            print(f"\033[{row};{col}H", end='', flush=True)
        except (ValueError, TypeError):
            self.push(y); self.push(x) # Restore stack
            raise TypeError(f"gotoxy requires two integers (Y X), got ({y}, {x})")

    def op_cls(self):
        """Clears the terminal screen."""
        # \033[2J clears entire screen
        # \033[H moves cursor to home position (1,1)
        print("\033[2J\033[H", end='', flush=True)

    # --- Метод парсинга _parse_token (С ИСПРАВЛЕННЫМ ПОРЯДКОМ) ---
    def _parse_token(self, current_line, current_col):
        start_pos = self.stream.tell()
        char = self.stream.read(1)
        if not char: return None, None, start_pos, current_line, current_col

        token_end_pos = self.stream.tell()
        line, col = current_line, current_col + 1
        value = None
        is_debug_on = getattr(self, 'DEBUG_MODE', False)

        # 1. Числа
        if char.isdigit() or char == '.' or char == '-':
            self.stream.seek(start_pos); col = current_col
            num_buffer = ""; prev_char_was_e = False; has_dot = False; has_e = False; is_first = True
            while True:
                 pcp = self.stream.tell(); c = self.stream.read(1); part_of_num = False
                 if c:
                      if c.isdigit(): part_of_num = True
                      elif c == '.' and not has_dot and not has_e: part_of_num=True; has_dot=True
                      elif c.lower() == 'e' and not has_e: part_of_num=True; has_e=True
                      elif c in '+-' and (is_first or prev_char_was_e): part_of_num = True
                 if not c or not part_of_num:
                     token_end_pos = self.stream.tell() - (1 if c else 0)
                     if c: self.stream.seek(pcp)
                     else: token_end_pos = self.stream.tell()
                     break
                 num_buffer += c; col +=1; prev_char_was_e = c.lower() == 'e'; is_first = False
            if num_buffer and num_buffer != '-' and num_buffer != '.':
                invalid_end = (num_buffer[-1].lower() == 'e' or (num_buffer[-1] in '+-') or (num_buffer[-1] == '.' and has_e))
                if not invalid_end:
                    if is_int(num_buffer): value = int(float(num_buffer))
                    elif is_float(num_buffer): value = float(num_buffer)
            if value is not None:
                if is_debug_on: print(f"DEBUG _parse_token: Found NUMBER '{num_buffer}' -> {value}, EndPos={token_end_pos}")
                return 'NUMBER', value, token_end_pos, line, col
            else: self.stream.seek(start_pos + 1); token_end_pos = self.stream.tell(); col = current_col + 1

        # 2. Строки
        if char == '"':
            str_val = ""; start_col_str = current_col + 1
            while True:
                nc = self.stream.read(1);
                if not nc: raise ValueError(f"Unclosed string starting at line {line}, col {start_col_str}")
                if nc == '\n': line += 1; col = 0
                else: col += 1
                token_end_pos = self.stream.tell()
                if nc == '"': break
                str_val += nc
            next_op_pos = self.stream.tell(); next_char = self.stream.read(1); is_assignment = False
            if next_char == '=':
                 if self.stack: is_assignment = True; token_end_pos = self.stream.tell()
                 else:
                     if next_char: self.stream.seek(next_op_pos)
            elif next_char: self.stream.seek(next_op_pos)
            if is_assignment:
                 if is_debug_on: print(f"DEBUG _parse_token: Found ASSIGN_VIA_EQ for '{str_val}', EndPos={token_end_pos}")
                 return 'ASSIGN_VIA_EQ', str_val, token_end_pos, line, col
            else:
                 if is_debug_on: print(f"DEBUG _parse_token: Found STRING '{str_val}', EndPos={token_end_pos}")
                 return 'STRING', str_val, token_end_pos, line, col

        # 3. Блоки кода
        if char == '{':
            block, nest, bl, bc = "", 1, line, current_col + 1
            while True:
                nc = self.stream.read(1);
                if not nc: raise ValueError(f"Unclosed code block {{}} ({bl},{bc})")
                if nc=='\n': line += 1; col = 0
                else: col += 1
                token_end_pos = self.stream.tell()
                if nc == '{': nest += 1
                elif nc == '}':
                    nest -= 1
                    if nest == 0: break
                block += nc
            full_block = "{" + block + "}"
            if is_debug_on: print(f"DEBUG _parse_token: Found BLOCK '{full_block[:30]}...', EndPos={token_end_pos}")
            return 'BLOCK', full_block, token_end_pos, line, col

        # 4. Массивы
        if char == '[':
            elem, nest, al, ac = "", 1, line, current_col + 1
            while True:
                nc = self.stream.read(1)
                if not nc: raise ValueError(f"Unclosed array [] ({al},{ac})")
                if nc=='\n': line += 1; col = 0; elem+=' '; continue
                else: col += 1
                token_end_pos = self.stream.tell()
                if nc == '[': nest += 1
                elif nc == ']':
                    nest -= 1
                    if nest == 0: break
                elem += nc
            try:
                np_str = '[' + elem.replace(';', ',') + ']'; safe_globals = {"__builtins__": None, "np": np}; arr = eval(np_str, safe_globals, {})
                arr = np.array(arr) if isinstance(arr,(list,tuple)) else arr
                if not isinstance(arr, np.ndarray): raise ValueError("Parsed array is not list/tuple/ndarray")
                if is_debug_on: print(f"DEBUG _parse_token: Found ARRAY (shape={getattr(arr,'shape','N/A')}), EndPos={token_end_pos}")
                return 'ARRAY', arr, token_end_pos, line, col
            except Exception as e: raise ValueError(f"Array parsing error '[{elem}]' ({al},{ac})") from e

        # *** НОВЫЙ ПОРЯДОК: СНАЧАЛА СЛОВА ***
        ident_start_col = current_col + 1
        token_end_pos = start_pos + 1 # Default: после char

        # 5. Идентификаторы / Слова
        if char.isalpha() or char == '_':
             self.stream.seek(start_pos); col = current_col
             ident = ""
             while True:
                 read_pos = self.stream.tell(); nc = self.stream.read(1)
                 if nc and (nc.isalnum() or nc == '_'):
                     ident += nc; col += 1; token_end_pos = self.stream.tell()
                 else:
                     if nc: self.stream.seek(read_pos)
                     else: token_end_pos = self.stream.tell()
                     break
             if not ident: raise SyntaxError(f"Invalid identifier starting with '{char}' ({line},{ident_start_col})")

             next_op_pos = self.stream.tell(); next_op = self.stream.read(1); assign = False; assign_op = ''
             if next_op == '→': assign = True; token_end_pos = self.stream.tell(); assign_op = '→'
             elif next_op == '=':
                  if self.stack: assign = True; token_end_pos = self.stream.tell(); assign_op = '='
                  else:
                      if next_op: self.stream.seek(next_op_pos)
             elif next_op: self.stream.seek(next_op_pos)

             if assign:
                 assign_type = 'ASSIGN_VIA_EQ' if assign_op == '=' else 'IDENTIFIER_ASSIGN'
                 if is_debug_on: print(f"DEBUG _parse_token: Found {assign_type} '{ident}', EndPos={token_end_pos}")
                 return assign_type, ident, token_end_pos, line, col
             elif ident in self.operations: # Алиас операции?
                 if is_debug_on: print(f"DEBUG _parse_token: Found OPERATOR (Word/Alias) '{ident}', EndPos={token_end_pos}")
                 return 'OPERATOR', ident, token_end_pos, line, col
             elif ident in self.variables: # Переменная/Константа/Слово?
                 value_in_var = self.variables[ident]
                 if isinstance(value_in_var, str) and value_in_var.startswith('{') and value_in_var.endswith('}'):
                      if is_debug_on: print(f"DEBUG _parse_token: Found USER_WORD '{ident}', EndPos={token_end_pos}")
                      return 'USER_WORD', value_in_var, token_end_pos, line, col
                 else: # Обычная переменная/константа
                      if is_debug_on: print(f"DEBUG _parse_token: Found VARIABLE (Word) '{ident}', EndPos={token_end_pos}")
                      return 'VARIABLE_WORD', value_in_var, token_end_pos, line, col
             else: # Неизвестное слово
                 raise NameError(f"Unknown identifier: '{ident}' ({line},{ident_start_col})")

        # 6. Односимвольные Операторы и Переменные (ПОСЛЕ СЛОВ)
        elif char in self.operations: # Включая '=' который не был частью слова
             if is_debug_on: print(f"DEBUG _parse_token: Found OPERATOR (Symbol) '{char}', EndPos={token_end_pos}")
             return 'OPERATOR', char, token_end_pos, line, col
        elif char in self.variables: # Только символьные константы
             if is_debug_on: print(f"DEBUG _parse_token: Found VARIABLE (Symbol) '{char}', EndPos={token_end_pos}")
             return 'VARIABLE_SYM', self.variables[char], token_end_pos, line, col
        # 7. Неизвестный символ
        else:
             raise SyntaxError(f"Unknown symbol/syntax starting with '{char}' ({line},{ident_start_col})")


    # --- Метод run (с логированием) ---
    def run(self, code):
        self.stream = io.StringIO(code)
        line, col = 1, 0
        # Управление отладкой
        self.DEBUG_MODE = getattr(self, 'DEBUG_MODE', False) # Наследовать или False
        # self.DEBUG_MODE = True # Раскомментировать для форсирования отладки
        if self.DEBUG_MODE: print(f"\n--- Running code block (len={len(code)}): {repr(code[:50])}...")
        start_token_line, start_token_col = 1, 1

        try:
            while True:
                start_token_pos = -1; start_token_line = line; start_token_col = col
                # --- Пропуск пробелов/комментариев ---
                while True:
                    current_pos = self.stream.tell(); char = self.stream.read(1);
                    if not char: break
                    if char == '\n': line += 1; col = 0; continue
                    if char.isspace(): col += 1; continue
                    if char == '/':
                        peek_char = self.stream.read(1)
                        if peek_char == '/': self.stream.readline(); line += 1; col = 0; continue
                        elif peek_char: self.stream.seek(self.stream.tell() - 1)
                    start_token_pos = current_pos; start_token_line = line; start_token_col = col + 1
                    self.stream.seek(start_token_pos); break
                if start_token_pos == -1: break # Конец файла

                if self.DEBUG_MODE: print(f"\nLOG: About to parse token @{start_token_pos}, L{start_token_line}, C{start_token_col}, Stack={list(self.stack)}")

                # --- Парсим токен ---
                token_type, token_value, token_end_pos, line_after, col_after = self._parse_token(start_token_line, start_token_col-1)
                if token_type is None: break

                # Сохраняем позицию конца токена для seek в конце итерации
                current_token_end_pos = token_end_pos

                line = line_after; col = col_after # Обновляем позицию для следующей итерации

                if self.DEBUG_MODE: print(f"LOG: Parsed -> Type='{token_type}', Val='{repr(token_value)[:50]}...', EndPos={token_end_pos}")

                # --- Обработка токена ---
                if token_type in ('NUMBER', 'STRING', 'BLOCK', 'ARRAY', 'VARIABLE_SYM', 'VARIABLE_WORD', 'IDENTIFIER_ASSIGN'):
                    self.push(token_value)
                    if self.DEBUG_MODE: print(f"  LOG: Action=PUSH, Value='{repr(token_value)[:30]}...'")
                elif token_type == 'ASSIGN_VIA_EQ':
                     var_name = token_value; self.push(var_name); self.op_store()
                     if self.DEBUG_MODE: print(f"  LOG: Action=ASSIGN_VIA_EQ, Var='{var_name}'")
                elif token_type == 'USER_WORD':
                    code_block_to_run = token_value
                    if self.DEBUG_MODE: print(f"  LOG: Action=EXECUTE_USER_WORD, Block='{token_value[:20]}...'")
                    self._execute_block(code_block_to_run)
                    # Позиция потока устанавливается после имени слова, а не после выполнения блока
                    if self.DEBUG_MODE: print(f"  LOG: Action=USER_WORD Done")
                elif token_type == 'OPERATOR':
                    op_key = token_value
                    if op_key in self.operations:
                        if self.DEBUG_MODE: print(f"  LOG: Action=EXECUTE_OP '{op_key}'")
                        self.operations[op_key]()
                        if self.DEBUG_MODE: print(f"  LOG: Action=OP '{op_key}' Done")
                    else: raise SyntaxError(f"Internal error: Unknown operator '{op_key}' ({start_token_line},{start_token_col})")
                else: raise SyntaxError(f"Internal error: Unknown token type '{token_type}' ({start_token_line},{start_token_col})")

                # Устанавливаем поток на конец обработанного токена
                self.stream.seek(current_token_end_pos)
                if self.DEBUG_MODE: print(f"  LOG: Stream seek to {current_token_end_pos} after {token_type}")

        except Exception as e: # Общий обработчик ошибок
             err_type = type(e).__name__; err_msg = str(e)
             location = f"(~line {start_token_line}, col {start_token_col})"
             full_msg = f"\nRuntime Error {location}: {err_type}: {err_msg}"
             if "Stack underflow for" in err_msg: full_msg = f"\nRuntime Error {location}: {err_msg}"
             stack_state_msg = f"Stack state: {list(self.stack)}" if 'Stack ' not in err_type else ""
             if stack_state_msg: print(full_msg, file=sys.stderr); print(stack_state_msg, file=sys.stderr)
             else: print(full_msg, file=sys.stderr)
             raise

# --- REPL Function ---
def start_repl(interpreter):
    print("SyMa REPL v0.9 (Light Logging & Parser Fix)")
    print("Введите 'quit', 'stack', 'vars', 'ops' или 'help'.")
    symbols_ops = sorted([k for k,v in interpreter.operations.items() if len(k)==1 and k not in interpreter.variables])
    aliases_ops = sorted([k for k,v in interpreter.operations.items() if len(k)>1])
    builtin_consts_sym = {k:v for k,v in interpreter.variables.items() if k in 'π𝑒𝑖∞⊤⊥'}
    builtin_consts_txt = {k:v for k,v in interpreter.variables.items() if k in ('pi','tau','e','j','inf','True','False')}

    while True:
        try:
            line_input = input("SyMa> ")
            if not line_input: continue
            cmd = line_input.strip().lower()
            if cmd == 'quit': break
            if cmd == 'stack': print("Stack:", list(interpreter.stack)); continue
            if cmd == 'vars':
                user_vars = {k:v for k,v in interpreter.variables.items() if k not in builtin_consts_sym and k not in builtin_consts_txt}
                print("Builtin Constants (Symbol):", builtin_consts_sym)
                print("Builtin Constants (Text):", builtin_consts_txt)
                if user_vars: print("User Variables:", user_vars)
                continue
            if cmd == 'ops' or cmd == 'operations':
                 print("Symbolic Ops:", " ".join(symbols_ops))
                 print("Word Aliases:", " ".join(aliases_ops))
                 continue
            if cmd == 'help':
                 print("Special commands: quit, stack, vars, ops, help")
                 print("Example: 5 3 add | pi 2 div sin")
                 print("Example: { dup mul } \"square\" store | 5 square")
                 print("Example: 10 \"x\" = | x 2 pow")
                 print("See documentation for more.")
                 continue
            line = line_input.replace(">=", "≥").replace("<=", "≤").replace("!=", "≠")
            line = re.sub(r'(?<=\s)\*(?=\s)', '×', line)
            line = re.sub(r'(?<=\s)/(?=\s)', '÷', line)
            if line != line_input: print(f"~ Executing: {line}")
            # Включаем/выключаем DEBUG_MODE для REPL
            # setattr(interpreter, 'DEBUG_MODE', True)
            interpreter.run(line)
            # setattr(interpreter, 'DEBUG_MODE', False)
            last_token = line.split()[-1] if line.split() else ''
            print_ops = ['▼', '▽', '✎', 'println', 'print', 'prints']
            if interpreter.stack and last_token not in print_ops:
                 print(f"-> {interpreter.peek()}")
        except IndexError as e: print(f"Stack Error: {e}", file=sys.stderr)
        except (TypeError, ValueError, NameError, np.linalg.LinAlgError, SyntaxError, RuntimeError, OverflowError, EOFError) as e:
            print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
            # Не печатаем стек здесь, т.к. run уже это делает при ошибке
            # print(f"Current stack: {list(interpreter.stack)}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected Error: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            # run уже напечатает стек при ошибке
            # print(f"Current stack: {list(interpreter.stack)}", file=sys.stderr)


# --- Основной блок ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            with open(filepath, 'r', encoding='utf-8') as f: code = f.read()
            main_interpreter = SyMaInterpreter()
            # *** Включите здесь для отладки файла ***
            # setattr(main_interpreter, 'DEBUG_MODE', True)
            main_interpreter.run(code)
        except FileNotFoundError: print(f"Error: File not found '{filepath}'", file=sys.stderr)
        except Exception as e: print(f"[Execution of file '{filepath}' failed]", file=sys.stderr)
    else:
        repl_interpreter = SyMaInterpreter()
        start_repl(repl_interpreter)