from typing import List


class Conflict:
    def __init__(
        self,
        ours: List[str],
        theirs: List[str],
        base: List[str],
        resolution: List[str] = None,
        resolution_kind: str = None,
    ):
        self.ours = ours
        self.theirs = theirs
        self.base = base
        self.resolution = resolution
        self.resolution_kind = resolution_kind
