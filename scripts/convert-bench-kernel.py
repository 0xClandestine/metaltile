#!/usr/bin/env python3
"""Convert #[bench_kernel(...)] #[kernel] to #[kernel(bench(...))] across all kernel files.

Transforms the old two-attribute pattern:

    #[bench_kernel(
        op = "unary",
        subop = "exp",
        class = Unary,
        tol = 1e-4,
    )]
    #[kernel]
    pub fn mt_exp<T>(...) { ... }

Into the unified single-attribute pattern:

    #[kernel(
        bench(
            op = "unary",
            subop = "exp",
            class = Unary,
            tol = 1e-4,
        )
    )]
    pub fn mt_exp<T>(...) { ... }
"""

import os
import re
import sys


def find_attr_end(lines: list[str], start_idx: int) -> int:
    """Find the line index where the attribute starting at start_idx ends.

    Attributes are of the form #[...]. We track bracket depth starting from
    the initial '[' at position start_idx and return the index of the line
    that contains the matching ']'.
    """
    depth = 0
    started = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == '[':
                depth += 1
                started = True
            elif ch == '(':
                depth += 1
            elif ch == ']':
                depth -= 1
            elif ch == ')':
                depth -= 1
        if started and depth == 0:
            return i
    return len(lines) - 1


def convert_file(filepath: str) -> bool:
    """Convert a single file. Returns True if any changes were made."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result: list[str] = []
    i = 0
    modified = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Detect #[bench_kernel( ... )]
        if stripped.startswith('#[bench_kernel('):
            attr_start = i
            attr_end = find_attr_end(lines, i)

            # Collect the full bench_kernel attribute block
            attr_lines = lines[attr_start : attr_end + 1]

            # Find the next non-empty line after the attribute
            next_line_idx = attr_end + 1
            while next_line_idx < len(lines) and lines[next_line_idx].strip() == '':
                next_line_idx += 1

            # Check if the next non-comment line is #[kernel]
            if (next_line_idx < len(lines)
                    and lines[next_line_idx].lstrip().startswith('#[kernel]')):

                # ── Convert! ─────────────────────────────────────

                # Determine indentation from the original bench_kernel line
                indent = line[:len(line) - len(line.lstrip())]
                inner_indent = indent + '    '

                # Strip the outer #[bench_kernel( ... )] wrapper to get the args.
                # The attribute is: #[bench_kernel(args)]
                # We need args.
                # Build the full attribute text (single string) to extract args.
                full_attr_text = ''.join(attr_lines)

                # Remove #[bench_kernel( prefix and )] suffix
                # The prefix is '#[bench_kernel('
                # The suffix is ')]'
                prefix = '#[bench_kernel('
                suffix = ')]'

                if full_attr_text.strip().startswith(prefix) and full_attr_text.strip().endswith(suffix):
                    # Extract the args portion
                    args_text = full_attr_text.strip()
                    args_text = args_text[len(prefix):-len(suffix)]
                    args_text = args_text.strip()
                else:
                    # Fallback: just strip the known prefix/suffix char by char
                    args_text = full_attr_text.strip()
                    # Remove leading '#[bench_kernel('
                    if args_text.startswith(prefix):
                        args_text = args_text[len(prefix):]
                    # Remove trailing ')]'
                    if args_text.endswith(suffix):
                        args_text = args_text[:-len(suffix)]
                    args_text = args_text.strip()

                # De-indent the args to the minimum indentation level,
                # then re-indent under the bench( ... ) wrapper.
                arg_lines = args_text.split('\n')
                if len(arg_lines) > 1:
                    # Find common leading whitespace
                    non_empty = [l for l in arg_lines if l.strip()]
                    if non_empty:
                        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
                        arg_lines = [l[min_indent:] if l.strip() else l for l in arg_lines]

                # Build the new unified attribute
                if len(arg_lines) == 1:
                    # Single-line: #[kernel(bench(op="unary", ...))]
                    arg_str = arg_lines[0].strip()
                    result.append(f'{indent}#[kernel(\n')
                    result.append(f'{inner_indent}bench({arg_str})\n')
                    result.append(f'{indent})]\n')
                else:
                    # Multi-line
                    result.append(f'{indent}#[kernel(\n')
                    result.append(f'{inner_indent}bench(\n')
                    for al in arg_lines:
                        if al.strip():
                            result.append(f'{inner_indent}{inner_indent}{al.lstrip()}\n')
                    result.append(f'{inner_indent})\n')
                    result.append(f'{indent})]\n')

                # Skip the #[kernel] line
                i = next_line_idx + 1

                # Preserve any blank lines between the attributes and the fn
                # Already handled naturally since we skip the #[kernel] line

                modified = True
                continue

        # Not a bench_kernel + kernel pair — pass through
        result.append(line)
        i += 1

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(result)
        print(f'  ✓ {filepath}')
    else:
        print(f'  - {filepath} (no changes)')

    return modified


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    # Find all relevant .rs files
    source_dirs = [
        'crates/metaltile-std/src',
        'crates/metaltile-std/tests',
        'crates/metaltile/src',
    ]

    files = []
    for srcdir in source_dirs:
        if os.path.isdir(srcdir):
            for root_dir, dirs, filenames in os.walk(srcdir):
                for f in filenames:
                    if f.endswith('.rs'):
                        files.append(os.path.join(root_dir, f))

    files.sort()
    print(f'Found {len(files)} .rs files to scan\n')

    total_modified = 0
    for filepath in files:
        if convert_file(filepath):
            total_modified += 1

    print(f'\nDone. {total_modified} file(s) modified.')

    if total_modified > 0:
        print('\n⚠  After running, verify the build with:')
        print('   cargo build --workspace')
        print('   cargo test -p metaltile-macros')


if __name__ == '__main__':
    main()