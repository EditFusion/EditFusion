package nju.merge.core;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.serializer.SerializerFeature;
import nju.merge.core.align.DeepMergeAligner;
import nju.merge.entity.ConflictFile;
import org.eclipse.jgit.diff.RawText;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.merge.MergeStrategy;
import org.eclipse.jgit.merge.RecursiveMerger;
import org.eclipse.jgit.merge.ThreeWayMerger;
import org.eclipse.jgit.revwalk.RevCommit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.StandardOpenOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class ConflictCollector {
    private static final Logger logger = LoggerFactory.getLogger(ConflictCollector.class);
    private final String projectName;
    private final String projectPath;
    private final String URL;
    private final String output;
    private Repository repository;

    private final Set<String> allowedExtentions;

    public ConflictCollector(String projectPath, String projectName, String url, String output, Set<String> allowedExtentions) {
        this.projectName = projectName;
        this.projectPath = projectPath;
        this.URL = url;
        this.output = output;
        this.allowedExtentions = allowedExtentions;
        try {
            repository = GitService.cloneIfNotExist(this.projectPath, URL);
        } catch (Exception e) {
            logger.error("Repository clone failed: {}", URL, e);
            try {
                // Append project name and URL to error_clone.txt
                Path errorPath = Paths.get(output, "error_clone.txt");
                Files.createDirectories(errorPath.getParent());
                Files.write(errorPath, Collections.singletonList(projectName + "," + URL), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException ioException) {
                logger.error("Failed to write error_clone.txt", ioException);
            }
        }
    }
    
    /**
     * Get base, ours, theirs, resolved and conflict versions of all source files with conflicts.
     * Conflict files contain conflict blocks.
     */
    public void process() throws Exception {
        if (repository == null) {
            return;
        }
        List<RevCommit> mergeCommits = GitService.getMergeCommits(repository); // All commits with two parents
        for (int i = 0; i < mergeCommits.size(); i++) {
            RevCommit commit = mergeCommits.get(i);
            if (i % 200 == 0) logger.info("Commit progress: {} out of {} merge commits, {}%", i, mergeCommits.size(), i * 100.0 / mergeCommits.size());
            mergeAndGetConflict(commit);
        }
    }

    private boolean isTargetFileType(String filename){
        // Split filename by "."
        String[] parts = filename.split("\\.");
        // Check if file has an extension and if it's allowed
        if (parts.length > 1) {
            String extension = parts[parts.length - 1];
            return this.allowedExtentions.contains(extension);
        }
        // No extension
        return false;
    }

    private void writeConflictFiles(String outputPath, List<ConflictFile> conflictFiles, String resolveHash) {
        Path jsonPath = Paths.get(outputPath, projectName, resolveHash, "conflictFilesMetadata.json");
        String jsonString = JSON.toJSONString(conflictFiles, SerializerFeature.PrettyFormat);
        try {
            writeContent(jsonPath, new String[]{jsonString});
        } catch (IOException e) {
            logger.error("Failed to write conflict files", e);
        }
        // If you want to write individual file versions, uncomment and adapt the following block:
        // for (ConflictFile conflictFile : conflictFiles) {
        //     String relativePath = conflictFile.filePath;
        //     try {
        //         Path basePath = Paths.get(outputPath, projectName, resolveHash, relativePath.replace("/", ":"));
        //         writeContent(basePath.resolve("base"), conflictFile.baseContent);
        //         writeContent(basePath.resolve("ours"), conflictFile.oursContent);
        //         writeContent(basePath.resolve("theirs"), conflictFile.theirsContent);
        //         writeContent(basePath.resolve("resolved"), conflictFile.resolvedContent);
        //     } catch (IOException e) {
        //         logger.error("Failed to write conflict file: {}", relativePath, e);
        //     }
        // }
    }
    
    private void writeContent(Path filePath, String[] content) throws IOException {
        // Ensure directory exists
        Files.createDirectories(filePath.getParent());
        // Write file content
        Files.write(filePath, Arrays.asList(content), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    private void mergeAndGetConflict(RevCommit resolve) {
        RevCommit ours = resolve.getParents()[0];
        RevCommit theirs = resolve.getParents()[1];

        ThreeWayMerger merger = MergeStrategy.RECURSIVE.newMerger(repository, true);
        try {
            if (!merger.merge(ours, theirs)) { // Conflicts found
                RecursiveMerger rMerger = (RecursiveMerger) merger;
                RevCommit base = (RevCommit) rMerger.getBaseCommitId();
                if (base == null) {
                    // Skip if base is null
                    logger.error("Base is null, {}, {}", projectName, resolve);
                    return;
                }
                List<ConflictFile> conflictFiles = new ArrayList<>();
                AtomicInteger processedFileCount = new AtomicInteger();
                rMerger.getMergeResults().forEach((file, result) -> {
                    if (isTargetFileType(file) && result.containsConflicts()) {
                        try {
                            String[] resolvedContent = GitService.getFileContent(this.repository, resolve, file);
                            String[][] contents = new String[3][];
                            for (int i = 0; i < 3; i++) {
                                contents[i] = new String(((RawText)result.getSequences().get(i)).getRawContent()).split("\n", -1);
                            }
                            ConflictFile conflictFile = new ConflictFile(
                                contents[0], contents[1], contents[2], 
                                null, resolvedContent, resolve.getName(), base.getName(), ours.getName(), theirs.getName(), file, projectName
                            );
                            DeepMergeAligner.getResolutions(conflictFile);
                            conflictFiles.add(conflictFile);
                        } catch (IOException e) {
                            // Most likely file not found
                            logger.warn("File with no corresponding resolved file: {}, {}, {}", projectName, resolve, file);
                        }
                    }
                });
                if (!conflictFiles.isEmpty()) {
                    // Write files
                    // output is the output directory, under which a projectName folder is created, then a folder for each resolved commit hash
                    // Each folder contains all conflict files for this merge, including base, ours, theirs, resolved
                    writeConflictFiles(output, conflictFiles, resolve.getName());
                }
            }
        } catch (IOException e) {
            logger.error("Failed to write conflict files", e);
        }
    }
}
