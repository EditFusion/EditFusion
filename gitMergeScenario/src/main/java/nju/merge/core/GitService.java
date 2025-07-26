package nju.merge.core;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.diff.RawText;
import org.eclipse.jgit.diff.RawTextComparator;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.ObjectLoader;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.merge.MergeAlgorithm;
import org.eclipse.jgit.merge.MergeFormatter;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.eclipse.jgit.treewalk.TreeWalk;
import org.eclipse.jgit.treewalk.filter.PathFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class GitService {

    private static final Logger logger = LoggerFactory.getLogger(GitService.class);

    public GitService(){}

    public static Repository cloneIfNotExist(String path, String url) throws Exception {
        File gitFolder = new File(path);
        Repository repo;
        if(gitFolder.exists()) {
            logger.info("Repository already exists locally: {}", path);
            FileRepositoryBuilder builder = new FileRepositoryBuilder();
            repo = builder.setGitDir(new File(gitFolder, ".git"))
                    .readEnvironment()
                    .findGitDir()
                    .build();
            return repo;
        } else{
            try {
                logger.info("Start cloning repository {}...", url);
                // If using SSH, ensure GIT_SSH is set correctly
                Git git = Git.cloneRepository()
                        .setURI(url)
                        .setDirectory(new File(path))
                        .setTimeout(600)
                        .call();

                logger.info("Clone completed");
                return git.getRepository();
            } catch (GitAPIException e) {
                logger.warn("Failed to clone repository {}", url);
                throw e;
            }
        }
    }

    public static String[] getMergedContent(String[] oursContent, String[] theirsContent, String[] baseContent) throws IOException {
    // The String[] here needs to use split("\n", -1) to ensure join correctly handles trailing newlines
    RawText base = new RawText((String.join("\n", baseContent) + "\n").getBytes(StandardCharsets.UTF_8));
    RawText ours = new RawText((String.join("\n", oursContent) + "\n").getBytes(StandardCharsets.UTF_8));
    RawText theirs = new RawText((String.join("\n", theirsContent) + "\n").getBytes(StandardCharsets.UTF_8));

    MergeAlgorithm mergeAlgorithm = new MergeAlgorithm();
    org.eclipse.jgit.merge.MergeResult<RawText> mergeResult = mergeAlgorithm.merge(
        RawTextComparator.DEFAULT, base, ours, theirs);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    MergeFormatter formatter = new MergeFormatter();

    // Use formatMergeDiff3 to output diff3 style
    formatter.formatMergeDiff3(
        out,
        mergeResult,
        Arrays.asList("BASE", "OURS", "THEIRS"),
        StandardCharsets.UTF_8
    );

    return out.toString(StandardCharsets.UTF_8).split("\n", -1);
    }

    public static List<RevCommit> getMergeCommits(Repository repository) throws Exception {
        // Collect all merge commits (commits with exactly two parents)
        logger.info("Collecting merge commits");

        List<RevCommit> commits = new ArrayList<>();

        try (RevWalk revWalk = new RevWalk(repository)) {
            for (Ref ref : repository.getRefDatabase().getRefs()) {
                revWalk.markStart(revWalk.parseCommit(ref.getObjectId()));
            }
            for (RevCommit commit : revWalk) {
                if (commit.getParentCount() == 2) {
                    commits.add(commit);
                }
            }
        } catch (IOException e) {
            logger.error("Error while collecting merge commits", e);
            logger.error("Repository: {}", repository.getDirectory().getAbsolutePath());
            logger.error("------------------------------------");
        }
        return commits;
    }

    public static String[] getFileContent(Repository repository, RevCommit commit, String filePath) throws IOException {
        // Create a TreeWalk object to traverse the tree in the commit
        try (TreeWalk treeWalk = new TreeWalk(repository)) {
            // Add the commit tree to the TreeWalk
            treeWalk.addTree(commit.getTree());
            treeWalk.setRecursive(true); // Enable recursive mode to find files
            treeWalk.setFilter(PathFilter.create(filePath));

            // Find the specified file and read its content
            if (!treeWalk.next()) {
                throw new IOException("File not found: " + filePath);
            }

            // Get file content
            ObjectId objectId = treeWalk.getObjectId(0);
            ObjectLoader loader = repository.open(objectId);
            byte[] fileContent = loader.getBytes();
            return new String(fileContent, StandardCharsets.UTF_8).split("\n");
        }
    }

}
