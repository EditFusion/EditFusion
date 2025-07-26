'''
This script processes results obtained from gitMergeScenario, reads the JSON files, replays the merge for the entire file, and recollects conflict chunks.
'''
# 类型提示放在这里，方便查看
class ConflictChunk:
    def __init__(self, m_start, m_end, a_content, b_content, 
                 o_content, r_content, label: str | None, chunk_idx):
        self.m_start = m_start
        self.m_end = m_end
        self.a_content: 'str' = a_content
        self.b_content: 'str' = b_content
        self.o_content: 'str' = o_content
        self.r_content: 'str' = r_content
        self.label = label
        self.chunk_idx = chunk_idx

    def to_dict(self):
        return {
            "m_start": self.m_start,
            "m_end": self.m_end,
            "a_content": self.a_content,
            "b_content": self.b_content,
            "o_content": self.o_content,
            "r_content": self.r_content,
            "label": self.label,
        }
    
    def getJSONstr(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)


class ConflictFile:
    def __init__(self, path, repo_url, file_a_content, file_b_content, file_o_content, file_r_content, file_m_content, commit_hash):
        self.path = path
        self.repo_url = repo_url
        self.file_a_content = file_a_content
        self.file_b_content = file_b_content
        self.file_o_content = file_o_content
        self.file_r_content = file_r_content
        self.file_m_content = file_m_content
        self.commit_hash = commit_hash
        self.conflict_chunks = []

    def add_conflict_chunk(self, conflict_chunk_obj):
        self.conflict_chunks.append(conflict_chunk_obj)

    def to_dict(self):
        return {
            "path": self.path,
            "repo_url": self.repo_url,
            "file_a_content": self.file_a_content,
            "file_b_content": self.file_b_content,
            "file_o_content": self.file_o_content,
            "file_r_content": self.file_r_content,
            "file_m_content": self.file_m_content,
            "conflict_chunks": [chunk.to_dict() for chunk in self.conflict_chunks],
        }
    
    def getJSONstr(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

class ConflictFileCollector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    @staticmethod
    def sample(output_dir, n, random_seed=0, label=None):
        cnt = 0
        # Randomly sample n Conflict chunks of the given label from all conflict files
        jsons = list(ConflictFileCollector.getAllJsonsUnder(output_dir))
        print(f"Found {len(jsons)} JSON files in {output_dir}")
        for json_file in jsons:
            with open(json_file) as f:
                data = json.load(f)
            for conflict_file in data:
                for chunk in conflict_file['conflict_chunks']:
                    if label is None or chunk['label'] == label:
                        if cnt >= n:
                            return
                        cnt += 1
                        yield chunk


    def collect(self):
        '''
        Returns an iterator, each iteration returns a ConflictFile object
        '''
        raise NotImplementedError
        
    def collect_in_batches(self, batch_size=10000):
        batch = []
        for conflict_file in self.collect():
            batch.append(conflict_file)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def collect_and_save(self, output_dir, batch_size=10000):
        output_dir = Path(output_dir)  # Ensure output_dir is a Path object
        output_dir.mkdir(parents=True, exist_ok=True)  # Automatically create directory and parents
        for i, batch in enumerate(self.collect_in_batches(batch_size)):
            with open(output_dir / f"{i}.json", 'w') as f:
                print(f"Saving batch {i} to {output_dir / f'{i}.json'}")
                json.dump([json.loads(x.getJSONstr()) for x in batch], f)
    
    @staticmethod
    def preprocessContent(content: str):
        return '' if content.strip() == '' else re.sub(r'\s+', ' ', content.strip() + '\n')
    
    @staticmethod
    def getLabel(a, b, o, r):
        r_processed = ConflictFileCollector.preprocessContent(r)
        a_processed = ConflictFileCollector.preprocessContent(a)
        b_processed = ConflictFileCollector.preprocessContent(b)
        o_processed = ConflictFileCollector.preprocessContent(o)
        if a_processed == b_processed:
            return "same modification, formatting maybe different"
        if r_processed == a_processed:
            return "A"
        if r_processed == b_processed:
            return "B"
        if r_processed == o_processed:
            return "O"
        if r_processed == a_processed + b_processed:
            return "AB"
        if r_processed == b_processed + a_processed:
            return "BA"

        r_lines = set(r.split('\n'))
        a_lines = set(a.split('\n'))
        b_lines = set(b.split('\n'))
        o_lines = set(o.split('\n'))
        for rl in r_lines:
            if (rl not in a_lines) and (rl not in b_lines) and (rl not in o_lines) and not rl.isspace():
                return 'newline'
        return 'mixline'

    @staticmethod
    def getAllJsonsUnder(dirPath: str):
        for root, _, files in os.walk(dirPath):
            for file in files:
                if(file.endswith(".json")):
                    yield os.path.join(root, file)
    
    @staticmethod
    def list2str(l):
        if l == [] or l == ['']:
            return ''
        return '\n'.join(l) + '\n'






# 1. Read JSON files, each is a list of conflictFile
# 2. For each conflictFile, get a_file_content, b_file_content, o_file_content, r_file_content
# 3. Write files, use git merge-file
# 4. Read conflict chunks, extract corresponding resolution
# 5. Overwrite file

import os
import re
import json
from pathlib import Path

def merge_file(a_content, b_content, o_content):
    """
    Start a new process, execute merge_file, and return the merged content
    params will be string
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Write files
        with open(os.path.join(tmpdirname, "a.txt"), "w") as f:
            f.write(a_content)
        with open(os.path.join(tmpdirname, "b.txt"), "w") as f:
            f.write(b_content)
        with open(os.path.join(tmpdirname, "o.txt"), "w") as f:
            f.write(o_content)
        # Execute merge-file
        os.system(f"git merge-file --diff3 {tmpdirname}/a.txt {tmpdirname}/o.txt {tmpdirname}/b.txt")
        # Read merged file
        with open(os.path.join(tmpdirname, "a.txt"), "r") as f:
            return f.readlines()
        

def process_conflict_file(conflict_file):
    """
    Process a single ConflictFile, extract conflict chunks, and return updated data
    """
    a_content: str = conflict_file["file_a_content"]
    b_content: str = conflict_file["file_b_content"]
    o_content: str = conflict_file["file_o_content"]
    r_content: str = conflict_file["file_r_content"]

    # Execute merge
    merged_lines_withEnding = merge_file(a_content, b_content, o_content)   # Each line ends with newline
    conflict_file['file_m_content'] = ''.join(merged_lines_withEnding)

    # Extract conflict chunks
    conflict_chunks = []
    in_conflict = False
    chunk = None
    current_content = ""  # Temporary storage for current chunk content
    chunk_idx = 0
    for i in range(len(merged_lines_withEnding)):
        line = merged_lines_withEnding[i]
        if line.startswith("<<<<<<<"):
            if in_conflict:
                # If already in a conflict chunk, file is invalid, return
                conflict_file["conflict_chunks"] = []
                return conflict_file
            in_conflict = True
            chunk = {
                "a_content": "",
                "b_content": "",
                "o_content": "",
                'm_start': i,
            }
            current_content = ""  # Initialize temporary variable
        elif line.startswith("|||||||"):
            if chunk is None:
                # If ||| appears first, file is invalid, return
                conflict_file["conflict_chunks"] = []
                return conflict_file
            chunk["a_content"] = current_content  # Save current chunk as a_content
            current_content = ""
        elif line.startswith("======="):
            if chunk is None:
                # If === appears first, file is invalid, return
                conflict_file["conflict_chunks"] = []
                return conflict_file
            chunk["o_content"] = current_content
            current_content = ""  # Clear temporary variable for b_content
        elif line.startswith(">>>>>>>"):
            if chunk is None:
                # If >>> appears first, file is invalid, return
                conflict_file["conflict_chunks"] = []
                return conflict_file
            chunk["b_content"] = current_content  # Save b_content
            chunk['chunk_idx'] = chunk_idx
            chunk_idx += 1
            chunk['m_end'] = i + 1
            in_conflict = False
            conflict_chunks.append(chunk)
            chunk = None
        elif in_conflict:
            current_content += line  # Accumulate current conflict chunk content

    ###### Solution extraction code below ######

    # Define minimal_unique_prefix function
    def minimal_unique_prefix(x, y):
        """
        Parameters
        ----------
        x : List[str]
            List to find prefix.
        y : List[str]
            Truth.
        Find the minimal unique prefix of x in y.
        Return the start index in y, or -1 if not found.
        """
        if not x:
            return -1
        # Initialize candidate indices where x[0] matches y
        candidates = {i for i, val in enumerate(y) if val == x[0]}
        offset = 0
        while len(candidates) > 1:
            offset += 1
            if offset == len(x):  # If offset reaches x length, no unique prefix found
                return -1
            to_remove = set()
            for idx in candidates:
                # Remove candidate if out of bounds or value does not match
                if idx + offset >= len(y) or y[idx + offset] != x[offset]:
                    to_remove.add(idx)
            candidates -= to_remove
        return candidates.pop() if candidates else -1
    
    # Prepare resolved_content, the content after resolution
    resolved_content_lines = list(map(lambda x: x + '\n', conflict_file["file_r_content"].splitlines()))        # Each line ends with newline

    len_after = len(resolved_content_lines) + 2  # Add 2 padding markers

    # Add padding markers before and after resolved_content
    truth_padded = ['<Begin Marker Here>'] + resolved_content_lines + ['<End Marker Here>']

    # Create reversed truth_padded for prefix matching
    reversed_truth_padded = truth_padded[::-1]

    for i in range(len(conflict_chunks) - 1, -1, -1):  # Reverse traversal for deletion
        cc = conflict_chunks[i]
        if 'm_start' not in cc:
            # If original text contains conflict chunk, skip this file
            conflict_file["conflict_chunks"] = []
            return conflict_file
        # Create subArr_eos for suffix matching
        subArr_eos = merged_lines_withEnding[cc['m_end']:] + ['<End Marker Here>']
        sffxIdx = minimal_unique_prefix(subArr_eos, truth_padded)  # Find suffix start index
        if sffxIdx == -1:
            # If no unique suffix found, delete current conflict chunk
            del conflict_chunks[i]
            continue

        # Create subArr_bos for prefix matching
        subArr_bos = merged_lines_withEnding[:cc['m_start']][::-1] + ['<Begin Marker Here>']
        prfxIdx = minimal_unique_prefix(subArr_bos, reversed_truth_padded)  # Find prefix start index
        if prfxIdx == -1:
            del conflict_chunks[i]
            continue

        # If condition met, extract solution
        if len_after - prfxIdx <= sffxIdx:
            start = len_after - prfxIdx
            end = sffxIdx
            cc['r_content'] = ''.join(truth_padded[start:end])
            cc['label'] = ConflictFileCollector.getLabel(cc['a_content'], cc['b_content'], cc['o_content'], cc['r_content'])
        else:
            del conflict_chunks[i]

    conflict_file["conflict_chunks"] = conflict_chunks
    return conflict_file



from tqdm import tqdm
from multiprocessing import Pool
import signal

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Child process ignores interrupt signal
    
def process_json_file(json_file_path):
    """
    Process a single JSON file, handle conflictFiles in parallel, and add tqdm progress bar.
    """
    with open(json_file_path, "r") as f:
        conflict_files = json.load(f)

    # Process conflictFiles in parallel
    cpus = os.cpu_count() - 8
    print(f"Using {cpus} CPUs")

    with tqdm(total=len(conflict_files), desc="Processing files", unit="file", dynamic_ncols=True) as pbar:
        def update(*args):
            """Update progress bar"""
            pbar.update()

        with Pool(cpus, initializer=initializer) as pool:
            try:
                # Use `imap_unordered` to process and update progress
                results = []
                for result in pool.imap_unordered(process_conflict_file, conflict_files):
                    results.append(result)
                    update()
            except KeyboardInterrupt:
                print('Manually stopped, exiting all processes')
                pool.terminate()

    return results

def process_directory(input_dir, output_dir):
    """
    Process all JSON files in a directory, replay merge and extract conflict chunks
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file_paths = list(input_dir.glob("*.json"))
    with tqdm(total=len(json_file_paths), desc="Processing JSON files", dynamic_ncols=True, unit="file") as outer_pbar:
        for json_file in json_file_paths:
            print(f"Processing {json_file}")
            updated_conflict_files = process_json_file(json_file)

            output_path = output_dir / json_file.name
            with open(output_path, "w") as f:
                json.dump(updated_conflict_files, f, indent=4)
            outer_pbar.update()

if __name__ == "__main__":
    input_dir = "/root/projects/dataset_collect_analysis/data_collect_analysis/output/100+stars_4GB-_multidev_org_lang"  # Replace with actual input directory
    output_dir = "/root/projects/dataset_collect_analysis/data_collect_analysis/output/100+stars_4GB-_multidev_org_lang"  # Replace with actual output directory

    process_directory(input_dir, output_dir)