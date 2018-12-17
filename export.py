import json
from pathlib import Path
import re
from typing import *

def is_code_cell(cell: dict):
    return (cell["cell_type"] == "code" and
            len(cell["source"]) > 0)

def is_md_cell(cell: dict):
    return cell["cell_type"] == "markdown" and len(cell["source"]) > 0

def indent(s: str):
    return " " * 4 + s

class MacroHandler:
    def __init__(self, header):
        self.header = header = header.strip()
        self.has_macro = False

        # ignore
        self.ignore = (header == "# ignore")

        # replace
        m = re.match("# replace\((.+),\s*(.+)\)", header)
        if m is not None:
            self.replace = lambda s: s.replace(m.group(1), m.group(2))
            self.has_macro = True
        else:
            self.replace = lambda s: s

        # uncomment
        if header == "# uncomment":
            self.uncomment = True
            self.has_macro = True
        else:
            self.uncomment = False

        # tests
        if re.match("# test_.+", header) is not None:
            self.is_test_func = True
            self.has_macro = True
        else:
            self.is_test_func = False

    def handle(self, lines: List[str]) -> str:
        if self.ignore: return ""
        if self.has_macro: lines = lines[1:]
        lines = [self.replace(line) for line in lines]
        if self.uncomment:
            lines = [line[2:] for line in lines]
        if self.is_test_func:
            test_func_name = self.header[2:]
            lines = [indent(line) for line in lines]
            lines = [f"def {test_func_name.strip()}():\n"] + lines
        return "".join(lines) + "\n\n"


def to_code_block(cell: dict):
    header = cell["source"][0]
    mh = MacroHandler(header)
    return mh.handle(cell["source"])

def export(data: dict, overwrite=False,
           script_dir=".", test_dir="."):
    # read library file name
    cells = data["cells"]

    if not is_md_cell(cells[0]):
        raise ValueError("No script filename provided")
    else:
        script_name = cells[0]["source"][0]

    contents = ""
    for i, cell in enumerate(cells[1:]):
        if is_code_cell(cell):
            contents += to_code_block(cell)
        elif is_md_cell(cell):
            if cell["source"][0].lower() == "# tests":
                break

    with open(Path(script_dir) / script_name, "wt") as f:
        f.write(contents.strip())

    tests = ""
    if not is_md_cell(cells[i+2]):
        raise ValueError("No test script name provided")
    else:
        test_script_name = cells[i+2]["source"][0]

    for i, cell in enumerate(cells[i+3:]):
        if is_code_cell(cell):
            tests += to_code_block(cell)

    with open(Path(test_dir) / f"{test_script_name}", "wt") as f:
        f.write(tests.strip())

if __name__ == "__main__":
    import sys
    data = json.load(open(sys.argv[1], "rt"))
    export(data, overwrite=True, script_dir="torchtable", test_dir="tests")
