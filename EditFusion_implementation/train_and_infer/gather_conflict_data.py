import os
from pathlib import Path
from collections import defaultdict
import json
from .utils.conflict_utils import Conflict
from tqdm import tqdm

script_path = Path(os.path.dirname(os.path.abspath(__file__)))

data_dir = script_path / "data" / "raw_data"
output_file = script_path / "data" / "gathered_data" / "2000repos.json"


print("Getting all conflict metadata JSON file paths...")


def getAllJsonsUnder(dirPath: str):
    for root, _, files in os.walk(dirPath):
        for file in files:
            if file.endswith(".json"):
                yield os.path.join(root, file)


jsonPaths = list(getAllJsonsUnder(str(data_dir)))


def collect_conflict_from_jsonPaths(paths, show_progress=True):
    kind_counter = defaultdict(int)
    fields = ["ours", "theirs", "base", "resolve"]
    results = []
    iter = tqdm(paths) if show_progress else paths
    for jsonPath in iter:
        with open(jsonPath) as jsonFile:
            jsonData = json.load(jsonFile)
            path, project_name = jsonData["path"], jsonData["project_name"]
            for chunk in jsonData["conflicting_chunks"]:

                if "resolve" not in chunk:
                    continue

                codes_origin = [
                    chunk["a_contents"],
                    chunk["b_contents"],
                    chunk["base_contents"],
                    chunk["resolve"],
                ]
                conflict_map = dict()
                conflict_map["path"] = path
                conflict_map["project_name"] = project_name

                if codes_origin[3] == None:
                    continue

                codes = list(
                    map(lambda s: str(s).strip("\n").split("\n"), codes_origin)
                )

                for i in range(4):
                    if codes[i] == [""]:
                        codes[i] = []
                    conflict_map[fields[i]] = codes[i]

                conflict = Conflict(
                    conflict_map[fields[0]],
                    conflict_map[fields[1]],
                    conflict_map[fields[2]],
                    conflict_map[fields[3]],
                )

                if conflict.base == conflict.ours or conflict.base == conflict.theirs:
                    continue

                if "resolution_kind" not in conflict_map:
                    ours, base, theirs, resolution = (
                        conflict.ours,
                        conflict.base,
                        conflict.theirs,
                        conflict.resolution,
                    )
                    if resolution == ours:
                        conflict_map["resolution_kind"] = "accept_ours"
                    elif resolution == theirs:
                        conflict_map["resolution_kind"] = "accept_theirs"
                    elif resolution == base:
                        conflict_map["resolution_kind"] = "accept_base"
                    elif resolution in [[""], []]:
                        conflict_map["resolution_kind"] = (
                            "delete_all"
                        )
                    elif any(
                        not (resolveline in ours + theirs + base)
                        for resolveline in resolution
                    ):
                        conflict_map["resolution_kind"] = "newline"
                    elif resolution == ours + theirs:
                        conflict_map["resolution_kind"] = "concat_ours_theirs"
                    elif resolution == theirs + ours:
                        conflict_map["resolution_kind"] = "concat_theirs_ours"
                    elif all(
                        resolveline in ours + theirs + base
                        for resolveline in conflict_map[fields[3]]
                    ):
                        conflict_map["resolution_kind"] = "mixline"
                    else:
                        conflict_map["resolution_kind"] = "others"

                kind_counter[conflict_map["resolution_kind"]] += 1
                results.append(conflict_map)
    return results, kind_counter



results, kind_counter = collect_conflict_from_jsonPaths(jsonPaths)
print(f"Collected {len(results)} conflict chunks.")
for k, v in kind_counter.items():
    print(f"{k}:{v} , {v/len(results)*100:.2f}%")

with open(output_file, "w") as f:
    json.dump(results, f)
