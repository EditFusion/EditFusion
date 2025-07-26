package nju.merge.entity;

import java.util.Optional;

public class ConflictChunk {
    public String[] base;
    public String[] ours;
    public String[] theirs;
    public String[] resolution; // 可选的解决方案
    public String label; // 可选标签
    public String resolvedCommitHash; // 维护合并场景的哈希值
    public String repositoryName; // 来源的仓库
    public String filePath; // 冲突块所在的文件路径
    public int startLine; // 冲突块的起始行
    public int endLine; // 冲突块的结束行(不包括)

    public ConflictChunk() {}

    @Override
    public String toString() {
        return "ConflictChunk{" +
                "base='" + base + '\'' +
                ", ours='" + ours + '\'' +
                ", theirs='" + theirs + '\'' +
                ", resolution=" + resolution +
                ", label=" + label +
                ", resolvedCommitHash='" + resolvedCommitHash + '\'' +
                ", repositoryName='" + repositoryName + '\'' +
                ", filePath='" + filePath + '\'' +
                '}';
    }
}