#!/usr/bin/env python3
"""
Reformat #[kernel(bench(...))] attribute blocks so that indentation is
always relative to the opening `#[kernel(` line:

  base + 0  →  #[kernel(
  base + 4  →      bench(
  base + 8  →          key=value,
  base + 4  →      )
  base + 0  →  )]

This is necessary for attributes inside macro_rules! arms where
cargo fmt / rustfmt cannot reach due to #[rustfmt::skip].

Usage:  python3 scripts/fmt_kernel_attrs.py
        python3 scripts/fmt_kernel_attrs.py path/to/file.rs   # single file
"""

import re
import sys
from pathlib import Path


def reformat_kernel_block(lines_in_block: list[str], base_indent: str) -> list[str]:
    """
    Re-indent a collected #[kernel(...)] block.
    lines_in_block[0] is the '#[kernel(' line itself (unchanged).
    Subsequent lines are re-indented based on paren depth.
    """
    result = [lines_in_block[0]]
    depth = 1  # inside the outer `(` of #[kernel(

    for raw in lines_in_block[1:]:
        stripped = raw.strip()

        if not stripped:
            result.append('')
            continue

        # Lines starting with ')' or ')]' close before indenting.
        starts_close = stripped[0] == ')'

        # Determine the indent for this line.
        effective_depth = (depth - 1) if starts_close else depth
        indent = base_indent + '    ' * effective_depth
        result.append(indent + stripped)

        # Update depth: count raw '(' and ')' — not '['/']'.
        opens  = stripped.count('(')
        closes = stripped.count(')')
        depth += opens - closes

    return result


def reformat_file(path: Path) -> bool:
    """Return True if the file was modified."""
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=False)

    out: list[str] = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        # Match a line that is purely `SPACES #[kernel(`
        m = re.match(r'^(\s*)#\[kernel\(\s*$', line)
        if m:
            base_indent = m.group(1)
            # Collect the full attribute block until depth reaches 0.
            block_raw = [line]
            depth = 1
            i += 1
            while i < len(lines) and depth > 0:
                l = lines[i]
                block_raw.append(l)
                depth += l.count('(') - l.count(')')
                i += 1

            # Reformat the block.
            block_new = reformat_kernel_block(block_raw, base_indent)

            if block_new != block_raw:
                changed = True
            out.extend(block_new)
        else:
            out.append(line)
            i += 1

    if changed:
        # Preserve trailing newline behaviour.
        new_text = '\n'.join(out)
        if text.endswith('\n') and not new_text.endswith('\n'):
            new_text += '\n'
        path.write_text(new_text, encoding='utf-8')
        return True
    return False


def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        root = Path(__file__).parent.parent
        targets = list(root.rglob('crates/**/*.rs'))

    modified = 0
    for path in sorted(targets):
        if reformat_file(path):
            print(f'reformatted  {path}')
            modified += 1

    print(f'\n{modified} file(s) updated.')


if __name__ == '__main__':
    main()
