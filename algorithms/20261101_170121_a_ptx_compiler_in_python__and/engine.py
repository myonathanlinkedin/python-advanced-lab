import re
from typing import List, Union, Tuple, Dict
from .types import Kernel, Instruction, Label


# ----------------------------------------------------------------------
# Parsing utilities
# ----------------------------------------------------------------------
_LABEL_RE = re.compile(r'^\s*([A-Za-z_][\w]*)\s*:\s*(?:#.*)?$')
_ENTRY_RE = re.compile(r'^\s*\.entry\s+([A-Za-z_][\w]*)\s*\{\s*(?:#.*)?$')
_END_RE = re.compile(r'^\s*\}\s*(?:#.*)?$')
_INSTR_RE = re.compile(
    r'''^\s*
        (?P<opcode>[A-Za-z_][\w]*)
        (?:\.(?P<type>[A-Za-z0-9_]+))?      # optional .type suffix
        \s+
        (?P<operands>[^;#]+)               # operands up to ';' or comment
        ;?\s*(?:#.*)?$                     # optional trailing ';' and comment
    ''',
    re.VERBOSE,
)


def _strip_comments(line: str) -> str:
    """Remove everything after a '#' character."""
    return line.split('#', 1)[0].rstrip()


def parse_ptx(source: str) -> Kernel:
    """
    Parse a tiny subset of PTX source code into a Kernel object.
    Supported constructs:
        - .entry <name> { ... }
        - labels: <label>:
        - instructions: opcode[.type] operand1, operand2, ...;
    Raises:
        ValueError: on syntax errors or unsupported constructs.
    """
    lines = source.splitlines()
    in_entry = False
    kernel_name = ''
    body: List[Union[Label, Instruction]] = []

    for raw_line in lines:
        line = _strip_comments(raw_line).strip()
        if not line:
            continue

        if not in_entry:
            m = _ENTRY_RE.match(line)
            if m:
                kernel_name = m.group(1)
                in_entry = True
                continue
            else:
                raise ValueError(f"Expected .entry declaration, got: {raw_line!r}")

        # Inside entry block
        if _END_RE.match(line):
            in_entry = False
            break

        # Label?
        m = _LABEL_RE.match(line)
        if m:
            body.append(Label(name=m.group(1)))
            continue

        # Instruction
        m = _INSTR_RE.match(line)
        if m:
            opcode = m.group('opcode')
            type_suffix = m.group('type')
            operands_raw = m.group('operands')
            operands = tuple(op.strip() for op in operands_raw.split(',') if op.strip())
            body.append(Instruction(opcode=opcode,
                                    type_suffix=type_suffix,
                                    operands=operands,
                                    raw=raw_line.rstrip()))
            continue

        raise ValueError(f"Unrecognised line inside kernel: {raw_line!r}")

    if in_entry:
        raise ValueError("Kernel block not closed with '}'")
    if not kernel_name:
        raise ValueError("No kernel entry found")
    return Kernel(name=kernel_name, body=body)


# ----------------------------------------------------------------------
# Simple compilation to a deterministic byte representation
# ----------------------------------------------------------------------
# Mapping of opcodes to a single-byte identifier (arbitrary but stable)
_OPCODE_MAP: Dict[Tuple[str, str | None], int] = {
    ('add', 's32'): 0x01,
    ('sub', 's32'): 0x02,
    ('mul', 's32'): 0x03,
    ('bra', None): 0x10,
    ('ret', None): 0xFF,
    # Extendable for more opcodes
}


def _encode_operand(op: str) -> bytes:
    """
    Encode an operand into a 4‑byte little‑endian integer.
    For this toy compiler we interpret:
        - Registers: %r<number> -> number
        - Predicates: %p<number> -> number + 0x8000
        - Immediate integers: decimal literals
    Anything else raises ValueError.
    """
    if op.startswith('%r'):
        num = int(op[2:])
        return num.to_bytes(4, 'little', signed=False)
    if op.startswith('%p'):
        num = int(op[2:]) + 0x8000
        return num.to_bytes(4, 'little', signed=False)
    # Immediate integer (e.g., 42)
    if re.fullmatch(r'-?\d+', op):
        num = int(op)
        return num.to_bytes(4, 'little', signed=True)
    raise ValueError(f"Unsupported operand format: {op!r}")


def compile_kernel(kernel: Kernel) -> bytes:
    """
    Compile a Kernel into a deterministic binary blob.
    The format is a simple sequence:
        [opcode-byte][operand-count][operand-bytes...]
    Labels are resolved to byte offsets during a two‑pass process.
    """
    # First pass: compute size of each instruction (without label resolution)
    instr_offsets: List[int] = []   # byte offset of each instruction in final blob
    current_offset = 0
    for item in kernel.body:
        if isinstance(item, Instruction):
            key = (item.opcode, item.type_suffix)
            if key not in _OPCODE_MAP:
                raise ValueError(f"Unsupported instruction: {item.opcode}{'.' + item.type_suffix if item.type_suffix else ''}")
            opcode_byte = _OPCODE_MAP[key]
            operand_count = len(item.operands)
            # opcode (1) + operand count (1) + each operand (4 bytes)
            size = 1 + 1 + operand_count * 4
            instr_offsets.append(current_offset)
            current_offset += size
        else:
            # Labels do not emit bytes
            instr_offsets.append(current_offset)

    # Second pass: emit bytes, resolving branch targets
    label_to_offset: Dict[str, int] = {}
    # Populate label offsets (they point to the next instruction's offset)
    for item, offset in zip(kernel.body, instr_offsets):
        if isinstance(item, Label):
            label_to_offset[item.name] = offset

    out = bytearray()
    for item in kernel.body:
        if isinstance(item, Instruction):
            key = (item.opcode, item.type_suffix)
            opcode_byte = _OPCODE_MAP[key]
            out.append(opcode_byte)
            out.append(len(item.operands))
            for op in item.operands:
                if item.opcode == 'bra' and item.type_suffix is None:
                    # Operand is a label name; replace with absolute byte offset
                    target_label = op
                    if target_label not in label_to_offset:
                        raise ValueError(f"Undefined label '{target_label}' in branch")
                    target_offset = label_to_offset[target_label]
                    out.extend(target_offset.to_bytes(4, 'little', signed=False))
                else:
                    out.extend(_encode_operand(op))
        # Labels emit nothing
    return bytes(out)


# ----------------------------------------------------------------------
# Hang detection (static analysis)
# ----------------------------------------------------------------------
def detect_hang(kernel: Kernel) -> bool:
    """
    Very coarse static analysis to detect kernels that contain an
    unconditional infinite loop without a reachable 'ret' instruction.
    Heuristic:
        - Find a label L such that the only instruction that can reach L
          after L is a 'bra L' (self‑loop) and there is no 'ret' before it.
        - If such a pattern exists, return True.
    This is *not* a full proof of non‑termination but suffices for the demo.
    """
    # Map label name -> index in body list
    label_index: Dict[str, int] = {}
    for idx, item in enumerate(kernel.body):
        if isinstance(item, Label):
            label_index[item.name] = idx

    # Find all 'bra' instructions and their target labels
    for idx, item in enumerate(kernel.body):
        if isinstance(item, Instruction) and item.opcode == 'bra':
            target = item.operands[0] if item.operands else ''
            if target not in label_index:
                continue  # undefined label will be caught later in compilation
            target_idx = label_index[target]

            # Self‑loop detection
            if target_idx == idx + 1:  # label immediately before the branch
                # Ensure there is no 'ret' before this branch in the same block
                # Scan backwards until previous label or start
                has_ret = False
                scan_idx = idx - 1
                while scan_idx >= 0:
                    prev = kernel.body[scan_idx]
                    if isinstance(prev, Label):
                        break
                    if isinstance(prev, Instruction) and prev.opcode == 'ret':
                        has_ret = True
                        break
                    scan_idx -= 1
                if not has_ret:
                    return True

    # No obvious infinite loop pattern found
    return False
