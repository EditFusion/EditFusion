package nju.merge.client;

import nju.merge.core.ConflictCollector;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Client {


//    private static String workdir = "";
//    private static String reposDir = workdir + "/repos";   // store all the repos
//    private static String outputDir = workdir + "/output";
//    private static String repoList = workdir + "/list.txt";
    private static final Logger logger = LoggerFactory.getLogger(Client.class);
    public static final Set<String> allowedExtensions = new HashSet<>(Arrays.asList(
            "py",    // Python
            "js",    // JavaScript
            "ts",    // TypeScript
            "go",    // Go
            "java",  // Java
            "cpp",   // C++
            "c",     // C
            "h",     // C Header
            "hpp",   // C++ Header
            "rb",    // Ruby
            "php",   // PHP
            "cs",    // C#
            "swift", // Swift
            "rs",    // Rust
            "m",     // Objective-C
            "mm"     // Objective-C++
    ));

    public static void addReposFromText(String txtPath, Map<String, String> repos) throws IOException {
        Path path = Paths.get(txtPath);
        List<String> lines = Files.readAllLines(path);
        lines.forEach(line -> {
            String[] args = line.split(",");
            repos.put(args[0].strip(), args[1].strip());
        });
    }

    public static void deleteRepo(String path2del) {
        try {
            FileUtils.deleteDirectory(new File(path2del));
        } catch (IOException e) {
            logger.error("path-to-delete is not a directory: {}", path2del, e);
        }
    }

    public static void collectMergeConflict(String repoPath, String projectName, String url, String output, Set<String> allowedExtensions) {
        try {
            ConflictCollector collector = new ConflictCollector(repoPath, projectName, url, output, allowedExtensions);
            collector.process();
        } catch (Exception e) {
            logger.error("收集遇到问题：{}", repoPath, e);
        }
        deleteRepo(repoPath);       // storage is limited, delete the repo after processing
    }
}
