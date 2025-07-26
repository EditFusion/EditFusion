package nju.merge.client;

import java.io.File;
import java.util.concurrent.ExecutionException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import nju.merge.core.GitService;
import nju.merge.core.align.DeepMergeAligner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.MergeCommand;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.junit.jupiter.api.Test;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;

import nju.merge.entity.ConflictFile;
import java.io.File;
import java.io.FileReader;
import java.util.List;


import nju.merge.utils.PathUtils;

public class ClientTest {
    private static final String workdir = "collect_output/";
    private static final String reposDir = workdir + "repos";   // store all the repos
    private static final String outputDir = workdir + "output";
    private static final Logger logger = LoggerFactory.getLogger(ClientTest.class);
    @Test
    public void collectTest() throws Exception {

        String list_file = workdir + "filtered_repos_100+stars_ssh_protocol.csv";

        HashMap<String, String> repos = new HashMap<>();
        Client.addReposFromText(list_file, repos);

        List<CompletableFuture<Void>> futures = new ArrayList<>();
    // Create a fixed-size thread pool
    int threadPoolSize = 4; // The number of threads you want
        ExecutorService executorService = Executors.newFixedThreadPool(threadPoolSize);
        AtomicInteger completedCnt = new AtomicInteger(0);

        repos.forEach((projectName, url) -> {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                String repoPath = PathUtils.getFileWithPathSegment(reposDir, projectName);
                String outputConflictPath = PathUtils.getFileWithPathSegment(outputDir, "conflictFiles");

                logger.info("Processing repo: {} on thread: {}", projectName, Thread.currentThread().getName());
                Client.collectMergeConflict(repoPath, projectName, url, outputConflictPath, Client.allowedExtensions);

                int completed = completedCnt.incrementAndGet();
                logger.info("Completed: {}/{}, {}%", completed, repos.size(), completed * 100.0 / repos.size());
            }, executorService);
            futures.add(future);
        });

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }


    public static List<ConflictFile> loadConflictFiles(String filePath) throws IOException {
    // Read file content as a string
        String content = new String(Files.readAllBytes(Paths.get(filePath)));

    // Parse JSON array into a list of ConflictFile
        return JSON.parseArray(content, ConflictFile.class);
    }

    private static List<File> getAllJsonFiles(File directory) {
        List<File> jsonFiles = new ArrayList<>();
        File[] files = directory.listFiles();

        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    jsonFiles.addAll(getAllJsonFiles(file)); // Recursively scan subdirectories
                } else if (file.isFile() && file.getName().endsWith(".json")) {
                    jsonFiles.add(file); // Add JSON file
                }
            }
        }
        return jsonFiles;
    }

    private static boolean deleteRecursively(File file) {
        if (file.isDirectory()) {
            File[] files = file.listFiles();
            if (files != null) {
                for (File f : files) {
                    deleteRecursively(f);
                }
            }
        }
        return file.delete();
    }

    @Test
    public void tmp() throws ExecutionException, InterruptedException {
        String directoryPath = "collect_output/output/conflictFiles";
        List<File> jsonFiles = getAllJsonFiles(new File(directoryPath));
        System.out.println("共有 " + jsonFiles.size() + " 个文件");

        List<CompletableFuture<Void>> futures = new ArrayList<>();
        ExecutorService executorService = Executors.newFixedThreadPool(40);
        AtomicInteger completedCnt = new AtomicInteger(0);
        for (File file : jsonFiles) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    List<ConflictFile> cfs = loadConflictFiles(file.getAbsolutePath());
                    
                    boolean processed = false;
                    for (ConflictFile cf : cfs) {
                        if (!cf.conflictChunks.isEmpty()) {
                            processed = true;
                            break;
                        }
                        cf.mergedContent = GitService.getMergedContent(cf.oursContent, cf.theirsContent, cf.baseContent);
                        DeepMergeAligner.getResolutions(cf);
                    }
                    if (processed) {
                        int completed = completedCnt.incrementAndGet();
                        if (completed % 100 == 0) System.out.println("Completed: " + completed + "/" + jsonFiles.size() + ", " + completed * 100.0 / jsonFiles.size() + "%");
                        return;
                    }

                    // Overwrite file with cfs
                    String jsonString = JSON.toJSONString(cfs);
                    Files.write(Paths.get(file.getAbsolutePath()), jsonString.getBytes());

                    // Delete all other files/directories in the same directory as file
                    File[] files = file.getParentFile().listFiles();
                    if (files != null) {
                        for (File f : files) {
                            if (!f.getName().equals(file.getName())) {
                                deleteRecursively(f);
                            }
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                int completed = completedCnt.incrementAndGet();
                if (completed % 100 == 0) System.out.println("Completed: " + completed + "/" + jsonFiles.size() + ", " + completed * 100.0 / jsonFiles.size() + "%");
            }, executorService);

            futures.add(future);
        }

    // Wait for all tasks to complete
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

    System.out.println("All files processed!");
    }


    @Test
    public void nativeRepoTest() {
        // String repoList = workdir + "/list.txt";

        String projectPath = "collect_output/native_test_repo_zsh/zsh";
        String projectName = "native_test_repo_zsh";
    String url = "git@github.com:zsh-users/zsh.git";            // Remember to set $GIT_SSH
        String output = "collect_output/output/test_repo";
        try {
            Client.collectMergeConflict(projectPath, projectName, url, output, Set.of("c"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void mergeTest() {
        String projectPath = "collect_output/tmp_repo";

        try {
            // Open the existing repository located at 'projectPath'
            File repoDir = new File(projectPath);
            Git git = Git.open(repoDir);
            Repository repo = git.getRepository();

            // Define the two commits you want to merge
            String oursHash = "b8bf0e4"; // 'ours' commit
            String theirsHash = "3e851";  // 'theirs' commit

            // Checkout to 'ours' commit
            git.checkout().setName(oursHash).call();

            // Resolve the ObjectIds for the commits
            ObjectId theirsId = repo.resolve(theirsHash);

            // Prepare the merge command
            MergeCommand merge = git.merge();
            merge.include(theirsId);               // Include 'theirs' commit in the merge

            // Execute the merge
            org.eclipse.jgit.api.MergeResult result = merge.call();

            // Handle the merge result
            if (result.getMergeStatus().isSuccessful()) {
                System.out.println("Merge was successful!");
            } else if (result.getMergeStatus().equals(org.eclipse.jgit.api.MergeResult.MergeStatus.CONFLICTING)) {
                System.out.println("Merge resulted in conflicts.");
                // retrieve and handle conflicts here if needed
                Map<String, int[][]> conflicts = result.getConflicts();
                if (conflicts != null) {
                    for (String path : conflicts.keySet()) {
                        System.out.println("Conflicts in file: " + path);
                    }
                }
            } else {
                System.out.println("Merge failed with status: " + result.getMergeStatus());
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void diff3Test() throws IOException {

        String baseContent = "a\nb\nc\nd\ne\nf\ng\nh\ni\n";
        String oursContent = "a\nb\n1\n2\n3\n6\nc\nd\ne\naaa\nf\nj\nk\nl\nm\n9\nxxx\ng\nh\nhhh\ni\n";
        String theirsContent = "a\nb\n4\n5\n6\nc\nd\ne\nf\n7\n8\n9\nxxx\ng\nh\ni\n";

        String[] out = GitService.getMergedContent(oursContent.split("\n", -1),
                theirsContent.split("\n", -1), baseContent.split("\n",-1));

        String mergedContent = String.join("\n", out);
        // String mergedContent = String.join("\n", out).replace("\n", "\\n");
        System.out.println(mergedContent);
    }
}
