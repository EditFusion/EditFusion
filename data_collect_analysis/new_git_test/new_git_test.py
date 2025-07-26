import random
from collections import defaultdict
import re
from tqdm import tqdm
import json
from git import Commit, Head, Repo, Git
import os

from numpy import delete
from pathlib import Path

script_path = Path(os.path.dirname(os.path.abspath(__file__)))

# data_dir = script_path / '..' / 'data_collect_analysis' / 'output' / '2000repos'
# data_dir = script_path / '..' / 'data_collect_analysis' / 'output' / 'mergebert_ts'
# data_dir = script_path / '..' / 'data_collect_analysis' / 'output' / 'mergebert_all_lang'
# data_dir = script_path / '..' / 'data_collect_analysis' / 'output' / '100+stars_4GB-_multidev_org_lang'
data_dir = script_path / '..' / 'data_collect_analysis' / 'output' / 'top50'
# Final dataset name
dataset_name = data_dir.stem
print(f"Start statistics for dataset {dataset_name}")


repo_path = Path(script_path / 'git_repo' / dataset_name)


def write_content_to_file(content: str | list[str], file_path: Path) -> None:
    # If content is a list, join it with '\n'
    if isinstance(content, list):
        if content in [[''], []]:
            content = ""            # Important: treat empty list as empty string
        else:
            content = '\n'.join(content) + '\n'
    file_path.write_text(content)


def remove_dir(dir_path: Path) -> None:
    # Remove directory if it exists
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_dir():
                remove_dir(item)
            else:
                item.unlink()
        dir_path.rmdir()

def clear_dir(repo_path: Path) -> None:
    # Remove existing repository
    if repo_path.exists():
        remove_dir(repo_path)

    # Initialize Git repository
    repo = Repo.init(repo_path)
    print(f"Initialized empty Git repository in {repo.git_dir}")

    # Create .gitignore file
    gitignore_path = repo_path / '.gitignore'
    write_content_to_file(['.DS_Store', '.vscode/', '.idea/'], gitignore_path)

    # Add and commit .gitignore file
    repo.index.add([str(gitignore_path)])
    repo.index.commit("Add .gitignore")
    print(f"Added and committed .gitignore file.")

    # Rename master branch to main
    if 'master' in repo.heads:
        repo.heads.master.rename('main')
    return repo

repo = clear_dir(repo_path)
tmpfile_path = Path(repo_path / 'tmp.txt')

# Path to compiled Git binary (update as needed for your environment)
new_git_path = '/root/projects/git/bin-wrappers/git'
log_path = script_path / 'log'
Git.git_exec_name = new_git_path
Git.refresh()
_git = Git(repo_path)



def create_branch(branch_name: str) -> Head:
    # Create a new branch from current commit
    _git.branch(branch_name)
    return repo.heads[branch_name]


def reset(delete_branch: str) -> None:
    _git.restore('.', '--staged')
    _git.restore('.', '--worktree')
    _git.checkout('main')
    # Delete branch if it exists
    if delete_branch in repo.heads:
        _git.branch('-D', delete_branch)
    # Reset main to the first commit
    no_parent_commit_generator = Commit.iter_items(
        repo=repo, rev="main",  max_parents=0)
    no_parent_commit = next(no_parent_commit_generator)
    _git.reset('--hard', str(no_parent_commit))


def commit_all(commit_message: str) -> None:
    # Track all files and commit
    _git.add('.')
    repo.index.commit(commit_message)


def replay(base_content, a_content, b_content):
    # Write base
    write_content_to_file(base_content, tmpfile_path)
    commit_all('base')
    # Create theirs branch
    theirs_branch = create_branch('theirs')
    _git.checkout(theirs_branch)
    # Write theirs
    write_content_to_file(b_content, tmpfile_path)
    commit_all('theirs')
    # Switch back to main
    _git.checkout('main')
    # Write ours
    write_content_to_file(a_content, tmpfile_path)
    commit_all('ours')
    # Merge theirs
    try:
        _git.merge('theirs')
    except Exception as e:
        return True  # merge conflict
    return False  # no conflict


def _log(save_name, kind_counter, kind_pseudo, kind_correct):
    def print_res(f):
        correct = sum(kind_correct.values())
        total = sum(kind_counter.values())
        print(f"Total = {total}", file=f)
        print(f"Pseudo-conflicts = {sum(kind_pseudo.values())}", file=f)
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
        print('Pseudo-conflict proportion by type:', file=f)
        print({kind: kind_pseudo[kind]/kind_counter[kind]
              * 100 for kind in kind_counter.keys()}, file=f)
        print('Pseudo-conflict accuracy by type:', file=f)
        print(pseudo_correct_ratio, file=f)
        print('Overall accuracy by type:', file=f)
        print(total_correct_ratio, file=f)

    if save_name is None:
        import sys
        print_res(sys.stdout)
        return
    with open(save_name, 'w') as f:
        print_res(f)

def preprocess(content: str) -> str:
    return '' if content.strip() == '' else re.sub(r'\s+', ' ', content.strip() + '\n')

reset(delete_branch='theirs')
# Begin statistics
kind_pseudo = defaultdict(int)
kind_counter = defaultdict(int)
kind_correct = defaultdict(int)

# Get all json files in data_dir
data_files = list(data_dir.glob('*.json'))
# Iterate over all files and read data
for file_path in tqdm(data_files):
    with open(file_path, 'r', encoding='utf-8') as f:
        cfs = json.load(f)
    # Iterate over all chunks and replay, record results
    for cf in tqdm(cfs):
        for chunk in cf['conflict_chunks']:
            a_content = chunk['a_content']
            b_content = chunk['b_content']
            base_content = chunk['o_content']

            kind_counter[chunk['label']] += 1
            has_conflict = replay(base_content, a_content, b_content)
            if has_conflict:
                reset(delete_branch='theirs')
                continue
            kind_pseudo[chunk['label']] += 1

            # If no conflict, read tmp.txt
            with open(tmpfile_path, 'r', encoding='utf-8') as f:
                result = f.read()
            if (preprocess(result) == preprocess(chunk['r_content'])):
                kind_correct[chunk['label']] += 1
            reset(delete_branch='theirs')
    repo = clear_dir(repo_path)

    _log(None, kind_counter, kind_pseudo, kind_correct)



_log(log_path / (f'{dataset_name}.log'), kind_counter, kind_pseudo, kind_correct)
# Plot and save results
def paint_new_git_result(kind_counter, kind_pseudo, kind_correct):
    import plotly.graph_objects as go
    fig = go.Figure()
    labels = list(kind_counter.keys())
    correct_pseudo = [kind_correct[label] for label in labels]
    wrong_pseudo = [kind_pseudo[label] - kind_correct[label] for label in labels]
    non_pseudo = [kind_counter[label] - kind_pseudo[label] for label in labels]

    fig.add_trace(go.Bar(
        x=labels,
        y=correct_pseudo,
        name='Correct pseudo-conflict'
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=wrong_pseudo,
        name='Incorrect pseudo-conflict',
        base=correct_pseudo
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=non_pseudo,
        name='Non-pseudo-conflict',
        base=[correct_pseudo[i] + wrong_pseudo[i] for i in range(len(labels))]
    ))

    fig.update_layout(
        barmode='stack',
        title=f'{dataset_name} pseudo-conflict statistics',
        xaxis_title='Conflict type',
        yaxis_title='Count',
        font=dict(
        family="Arial, Microsoft YaHei, SimHei",  # Specify multiple fonts in order
        size=14
    )
    )

    # Save as html file
    fig.write_html(log_path / f'{dataset_name}_new_git_result.html')

paint_new_git_result(kind_counter, kind_pseudo, kind_correct)
