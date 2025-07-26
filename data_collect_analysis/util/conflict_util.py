from typing import List
import json
from pathlib import Path

class Conflict:
    def __init__(self, ours: List[str], theirs: List[str], base: List[str], resolution: List[str] = None, resolution_kind: str = None):
        self.ours = ours
        self.theirs = theirs
        self.base = base
        self.resolution = resolution
        self.resolution_kind = resolution_kind

def conflict2file(conflict: Conflict, path: Path):
    def list2line(list: List[str]):
        return '\n'.join(list)
    with open(path / 'base.txt', 'w') as f:
        f.write(list2line(conflict.base))
    with open(path / 'ours.txt', 'w') as f:
        f.write(list2line(conflict.ours))
    with open(path / 'theirs.txt', 'w') as f:
        f.write(list2line(conflict.theirs))
    if conflict.resolution is not None:
        with open(path / 'resolution.txt', 'w') as f:
            f.write(list2line(conflict.resolution))