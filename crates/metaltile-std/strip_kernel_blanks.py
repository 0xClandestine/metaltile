#!/usr/bin/env python3
"""
Remove blank lines from inside #[kernel] function bodies in src/ffai/*.rs
and src/mlx/**/*.rs.

Usage:
  python3 strip_kernel_blanks.py [--dry-run] [files...]

  If no files are given, processes all src/ffai/*.rs and src/mlx/**/*.rs.
  With --dry-run, prints diffs without modifying files.
"""

import difflib
import re
import sys
from pathlib import Path


def find_block_end(text: str, open_pos: int) -> int | None:
    """Return index just after the matching '}' for '{' at open_pos."""
    assert text[open_pos] == '{'
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        if text[i:i+2] == '//':
            eol = text.find('\n', i)
            i = eol + 1 if eol != -1 else n
            continue
        if text[i:i+2] == '/*':
            end = text.find('*/', i + 2)
            i = end + 2 if end != -1 else n
            continue
        if text[i] == '"':
            i += 1
            while i < n:
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def strip_blank_lines_in_kernel_bodies(text: str) -> str:
    """Remove blank lines from inside every #[kernel] fn body."""
    result = []
    pos = 0

    # Match the opening brace of a #[kernel] pub fn body.
    # Pattern: #[kernel] (optional whitespace/newlines) pub [unsafe] fn name<...>(...) {
    pat = re.compile(
        r'#\[kernel\]'
        r'(?:\s*#\[[^\]]*\])*'          # any extra attrs (#[cfg...] etc.)
        r'\s*pub\s+(?:unsafe\s+)?fn\s+\w+'
        r'(?:<[^>]*>)?'                  # optional generics
        r'\s*\([^)]*(?:\([^)]*\)[^)]*)*\)'  # params (handles nested parens)
        r'\s*\{',
        re.DOTALL,
    )

    for m in pat.finditer(text):
        brace_pos = text.rindex('{', m.start(), m.end())
        body_end = find_block_end(text, brace_pos)
        if body_end is None:
            continue

        # Emit text up to and including the opening brace
        result.append(text[pos:brace_pos + 1])

        # Strip blank lines from body (between '{' and the closing '}')
        body = text[brace_pos + 1 : body_end - 1]
        # Remove lines that are entirely whitespace
        stripped = re.sub(r'\n[ \t]*(?=\n)', '', body)
        result.append(stripped)

        # Closing brace
        result.append(text[body_end - 1])
        pos = body_end

    result.append(text[pos:])
    return ''.join(result)


def process(path: Path, dry_run: bool) -> bool:
    original = path.read_text(encoding='utf-8')
    result = strip_blank_lines_in_kernel_bodies(original)
    if result == original:
        return False
    if dry_run:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            result.splitlines(keepends=True),
            fromfile=str(path),
            tofile=f'{path} (modified)',
            n=2,
        )
        sys.stdout.writelines(diff)
        return True
    path.write_text(result, encoding='utf-8')
    return True


def main() -> None:
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    args = [a for a in args if a != '--dry-run']

    if args:
        files = [Path(a) for a in args]
    else:
        here = Path(__file__).parent
        files = sorted((here / 'src' / 'ffai').glob('*.rs'))
        files += sorted((here / 'src' / 'mlx').rglob('*.rs'))

    changed = 0
    for path in files:
        if not path.exists():
            print(f'Not found: {path}', file=sys.stderr)
            continue
        if process(path, dry_run):
            changed += 1
            action = 'would modify' if dry_run else 'modified'
            print(f'  {action}: {path.name}')

    print(f"\n{'Would modify' if dry_run else 'Modified'} {changed}/{len(files)} files.")


if __name__ == '__main__':
    main()
