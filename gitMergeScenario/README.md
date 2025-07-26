# Usage

- The `output` directory in the client contains the paths for conflict files and conflict tuple outputs. The `gitPath` is used for downloading and storing git repositories; you can modify these paths as needed.
- Repositories to be analyzed can be manually set following the format in `addSimpleRepo`: (project name, remote URL). After configuration, simply run the main method in the client.
- Due to network issues in China, the JGit clone command may fail. Please download repositories manually to your local machine if needed.

# Functionality

## Collecting Conflict Files
Traverse the git history to collect conflict files. These are output to `output/conflictFiles` and stored in the format: `commitId/filepath/filename.java/conflict files`.

## Collecting Conflict Tuples
Traverse the files collected in the previous step and extract conflict tuples from conflict markers. These are stored as `projectname.json` in `output/mergeTuples`.

## Statistics on Conflict Tuples
Statistical results are printed to standard output, for example:

![Example Output](https://user-images.githubusercontent.com/61650772/178206331-3eb4b3ca-4567-42d8-8387-21c96a6bd8ef.png)

# Input

- The root directory should contain a `list.txt` file, comma-separated, listing the repositories to analyze by project name and URL:
  ```
  # Example
  junit4,tmpurl
  spring-boot,tmp
  ```
- The `repos` directory in the root should contain the actual repositories:
  ```
  repos
  ├── junit4
  └── spring-boot
  ```

# Output

- The `output` directory in the root records the results of dataset collection:
  ```
  output
  ├── conflictFiles           # Collected files containing conflicts (e.g., conflict.java with conflict blocks)
  │   ├── junit4
  │   └── spring-boot
  ├── filteredTuples          # Filtered merge tuples
  │   ├── lackOfResolution    # Missing resolution
  │   ├── mixLine             # Mixed lines
  │   └── outOfVocabulary     # Newly added statements
  └── mergeTuples             # All merge tuples
      ├── junit4.json
      └── spring-boot.json
  ```