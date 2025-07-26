import random
from collections import defaultdict
from email.mime import base
import re
from shlex import join
from typing import List
from tqdm import tqdm
import json
from git import Commit, Head, Repo, Git
import os
from util.conflict_util import Conflict, conflict2file
from pathlib import Path
from util.tokenizer_util import encode, decode

script_path = Path(os.path.dirname(os.path.abspath(__file__)))
log_path = Path(script_path / '..' / 'log' / 'ES_unresolvable_self_collected_most_50_token_conflict_test2.log')

repo_path = Path(script_path / ".." / "git_repo")
repo = Repo(repo_path)
tmpfile_path = Path(repo_path / 'tmp.txt')

_git = Git(repo_path)

no_parent_commit_generator = Commit.iter_items(
    repo=repo, rev="main",  max_parents=0)  # 找到 reachable 最早的 commit
no_parent_commit = next(no_parent_commit_generator)

file_path = script_path / ".." / 'output' / 'self_collected_most_50_unresolvable.json'
with open(file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)


def write_content_to_file(content: str | list[str], file_path: Path) -> None:
    # If content is a list, join it with '\n'
    if isinstance(content, list):
        # content = '\n'.join(content)
        if content in [[''], []]:
            content = ""            # ! Very important
        else:
            content = '\n'.join(content) + '\n'
    file_path.write_text(content)

def create_branch(branch_name: str) -> Head:
    # Create a new branch based on the current commit
    _git.branch(branch_name)
    return repo.heads[branch_name]


def reset(delete_branch: str) -> None:
    _git.restore('.', '--staged')
    _git.restore('.', '--worktree')
    _git.checkout('main')
    # Check if the branch to delete exists
    if delete_branch in repo.heads:
        # Delete branch
        _git.branch('-D', delete_branch)
    # Reset main to the first commit
    _git.reset('--hard', str(no_parent_commit))


def commit_all(commit_message: str) -> None:
    # Track all files
    _git.add('.')
    # Commit
    repo.index.commit(commit_message)


def replay(conflict: Conflict):
    # Write base
    write_content_to_file(conflict.base, tmpfile_path)
    # 提交
    commit_all('base')
    # Create theirs branch
    theirs_branch = create_branch('theirs')
    _git.checkout(theirs_branch)
    # Write theirs
    write_content_to_file(conflict.theirs, tmpfile_path)
    # 提交
    commit_all('theirs')
    # Switch back to main
    _git.checkout('main')
    # Write ours
    write_content_to_file(conflict.ours, tmpfile_path)
    # 提交
    commit_all('ours')
    # Merge theirs
    try:
        _git.merge('theirs')
    except Exception as e:
        return True # merge conflict
    return False # no conflict

def debug_view(conflict: Conflict):
    '''Write conflict to local file for diffviewer inspection'''
    # NOTE: The following path is absolute and may expose personal directory structure
    conflict2file(conflict, Path('/Users/foril/projects/conflict_resolve/test_with_vscode/tmp'))


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid warnings
############################# Start token-level conflict statistics #############################
random.seed(42)
random.shuffle(data)
kind_counter = defaultdict(int)
token_resolvable_counter = defaultdict(int)
correct_counter = defaultdict(int)


for conflict_dict in tqdm(data[:], dynamic_ncols=True):
    reset(delete_branch='theirs')
    # 1. Get conflict
    conflict = Conflict(conflict_dict['ours'], conflict_dict['theirs'],
                        conflict_dict['base'], conflict_dict['resolution'], conflict_dict['resolution_kind'])
    kind_counter[conflict.resolution_kind] += 1
    
    debug_view(conflict)

    def content_preprocess(content_list: List[str]):
        # !important
        if content_list in [[''], []]:
            return ""
        content_list = list(filter(lambda x: x != '', content_list))
        return '\n'.join(content_list) + '\n'   # TODO: Check correctness
    
    base_content = content_preprocess(conflict.base)
    ours_content = content_preprocess(conflict.ours)
    theirs_content = content_preprocess(conflict.theirs)
    resolution_content = content_preprocess(conflict.resolution)
    # 2. Tokenize ours, theirs, base, resolution
    def tokenize_to_str_list(content: str) -> List[str]:
        encoded_list = encode(content)
        decoded_strs = decode(encoded_list)
        return decoded_strs # type: ignore
    
    # 3. Each token is a line
    base_tokens = tokenize_to_str_list(base_content)
    ours_tokens = tokenize_to_str_list(ours_content)
    theirs_tokens = tokenize_to_str_list(theirs_content)
    resolution_tokens = tokenize_to_str_list(resolution_content)

    # 4. Replay to check for conflict
    tokenized_conflict = Conflict(ours_tokens, theirs_tokens, base_tokens, resolution_tokens, conflict.resolution_kind)
    debug_view(tokenized_conflict)
    has_conflict = replay(tokenized_conflict)
    if has_conflict:
        continue
    token_resolvable_counter[conflict.resolution_kind] += 1

    # 5. Compare if the merged result without conflict matches
    merged_content = tmpfile_path.read_text()
    joint_merged = ''.join(merged_content.split('\n'))
    joint_resolution = ''.join(resolution_tokens)
    # todo：比较时去除空行  在上面 preprocess 时已经去除了
    if joint_merged == joint_resolution:
        correct_counter[conflict.resolution_kind] += 1

def _log(save_name, kind_counter, kind_pseudo, kind_correct):
    with open(save_name, 'w') as f:
        correct = sum(kind_correct.values())
        total = sum(kind_counter.values())
        print(f"Total = {total}", file=f)
        print(f"Pseudo-conflict = {sum(kind_pseudo.values())}", file=f)
        print(f"Correct = {correct}", file=f)
        print(f"Pseudo-conflict accuracy = {correct/sum(kind_pseudo.values())*100}%", file=f)
        print(f"Overall accuracy = {correct/total*100}%", file=f)
        print('Count by type:', file=f)
        print(kind_counter, file=f)
        print('Pseudo-conflict count by type:', file=f)
        print(kind_pseudo, file=f)
        print('Correct count by type:', file=f)
        print(kind_correct, file=f)

        print('-' * 30, file=f)

        total_correct_ratio = {
            kind: kind_correct[kind]/kind_counter[kind]*100 for kind in kind_counter.keys()}
        pseudo_correct_ratio = {
            kind: kind_correct[kind]/kind_pseudo[kind]*100 if kind_pseudo[kind] != 0 else 0 for kind in kind_counter.keys()}
        print('Pseudo-conflict ratio by type:', file=f)
        print({kind: kind_pseudo[kind]/kind_counter[kind]
              * 100 for kind in kind_counter.keys()}, file=f)
        print('Pseudo-conflict accuracy by type:', file=f)
        print(pseudo_correct_ratio, file=f)
        print('Overall accuracy by type:', file=f)
        print(total_correct_ratio, file=f)

_log(log_path, kind_counter, token_resolvable_counter, correct_counter)
