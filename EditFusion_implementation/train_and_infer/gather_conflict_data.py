# deprecated
import os
from pathlib import Path
from collections import defaultdict
import json
from .utils.conflict_utils import Conflict
from tqdm import tqdm

script_path = Path(os.path.dirname(os.path.abspath(__file__)))

# fill here ####
data_dir = script_path / "data" / "raw_data"  # 包含仓库收集到的所有冲突数据的根文件夹
output_file = script_path / "data" / "gathered_data" / "2000repos.json"
################


print("获取所有冲突元信息 JSON 文件路径")


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
            # print(jsonFile.read())
            jsonData = json.load(jsonFile)
            path, project_name = jsonData["path"], jsonData["project_name"]
            for chunk in jsonData["conflicting_chunks"]:

                if "resolve" not in chunk:
                    continue  # 本次收集不包含没有 resolution 的冲突块

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
                    continue  # 本次收集不包含没有 resolution 的冲突块

                codes = list(
                    map(lambda s: str(s).strip("\n").split("\n"), codes_origin)
                )  # 去除前后的换行符，会导致一些空白冲突在数据集中只剩相同部分
                # 这里如果不 strip 的话，几乎所有 concat 都会被匹配到 mixline，综上，选择 strip 并且过滤空白符冲突 在比较时过滤所有空行影响
                # codes = list(map(lambda s: str(s).split("\n"), codes_origin))      # 去除前后的换行符，会导致一些空白冲突在数据集中只剩相同部分

                for i in range(4):
                    # 去除空行 '' 不会被去除，但 '\n', ' \n', '\n\n', ' ' 会被去除，换行符的冲突会被显示为全空冲突
                    # codes[i] = list(filter(lambda line: not(line == '' or line.isspace()), codes[i]))     # 过滤空行
                    if codes[i] == [""]:
                        codes[i] = []  # 过滤空行被匹配到编辑脚本的情况
                    conflict_map[fields[i]] = codes[i]

                conflict = Conflict(
                    conflict_map[fields[0]],
                    conflict_map[fields[1]],
                    conflict_map[fields[2]],
                    conflict_map[fields[3]],
                )

                # 过滤 CRLF/LF 冲突 以及 空白符冲突
                if conflict.base == conflict.ours or conflict.base == conflict.theirs:
                    continue

                if "resolution_kind" not in conflict_map:
                    ours, base, theirs, resolution = (
                        conflict.ours,
                        conflict.base,
                        conflict.theirs,
                        conflict.resolution,
                    )
                    if resolution == ours:  # accept ours
                        conflict_map["resolution_kind"] = "accept_ours"
                    elif resolution == theirs:  # accept theirs
                        conflict_map["resolution_kind"] = "accept_theirs"
                    elif resolution == base:  # accept base
                        conflict_map["resolution_kind"] = "accept_base"
                    elif resolution in [[""], []]:
                        conflict_map["resolution_kind"] = (
                            "delete_all"  # MergeBert 数据集中 resolution 是 None 的应该是 lack of resolution？
                        )
                    elif any(
                        not (resolveline in ours + theirs + base)
                        for resolveline in resolution
                    ):  # newline
                        conflict_map["resolution_kind"] = "newline"
                    elif resolution == ours + theirs:
                        # conflict2file(conflict, Path('/Users/foril/projects/conflict_resolve/test_with_vscode/tmp'))
                        conflict_map["resolution_kind"] = "concat_ours_theirs"
                    elif resolution == theirs + ours:
                        # conflict2file(conflict, Path('/Users/foril/projects/conflict_resolve/test_with_vscode/tmp'))
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
print(f"共收集到 {len(results)} 个冲突块")
for k, v in kind_counter.items():
    print(f"{k}:{v} , {v/len(results)*100:.2f}%")

# 保存结果
with open(output_file, "w") as f:
    json.dump(results, f)
