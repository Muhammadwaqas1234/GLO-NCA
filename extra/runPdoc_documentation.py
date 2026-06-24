import os

"""Generate HTML API docs with pdoc (https://pdoc.dev/docs/pdoc.html).

Modern pdoc (>=10) outputs HTML by default and writes to a directory given by
-o; the old pdoc3 flags '--html' / '--force' have been removed. Run from the
project root so the 'src' package is importable.
"""

def addFileToDocumentation(path, out_dir="docs"):
    os.system(f'pdoc {path} -o {out_dir}')

def main():
    # Document the whole src package into ./docs
    os.system('pdoc src -o docs')

if __name__ == '__main__':
    main()

