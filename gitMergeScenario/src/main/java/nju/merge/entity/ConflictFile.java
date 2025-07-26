package nju.merge.entity;

import java.util.ArrayList;
import java.util.List;

public class ConflictFile {
    public String[] baseContent;      // 基准版本的文件内容
    public String[] oursContent;      // 我方版本的文件内容
    public String[] theirsContent;    // 对方版本的文件内容
    public String[] mergedContent;    // 合并后的文件内容（可能包含冲突标记）
    public String[] resolvedContent;  // 解决冲突后的文件内容

    public String resolvedCommitHash;// 合并场景的提交哈希值
    public String baseHash;         // 基准版本的哈希值
    public String oursHash;         // 我方版本的哈希值
    public String theirsHash;       // 对方版本的哈希值

    public String filePath;         // 文件路径
    public String repositoryName;   // 来源的仓库
    public List<ConflictChunk> conflictChunks = new ArrayList<>(); // 冲突块

    public ConflictFile(String[] baseContent, String[] oursContent, String[] theirsContent, String[] mergedContent,
                        String[] resolvedContent, String resolvedCommitHash, String baseHash, String oursHash,
                        String theirsHash, String filePath, String repositoryName) {
        this.baseContent = baseContent;
        this.oursContent = oursContent;
        this.theirsContent = theirsContent;
        this.mergedContent = mergedContent;
        this.resolvedContent = resolvedContent;
        
        this.resolvedCommitHash = resolvedCommitHash;
        this.baseHash = baseHash;
        this.oursHash = oursHash;
        this.theirsHash = theirsHash;
        this.filePath = filePath;
        this.repositoryName = repositoryName;
    }

    public void addConflictChunk(String[] base, String[] ours, String[] theirs, int startLine, int endLine) {
        ConflictChunk cc = new ConflictChunk();
        cc.base = base;
        cc.ours = ours;
        cc.theirs = theirs;
        cc.startLine = startLine;
        cc.endLine = endLine;

        cc.repositoryName = this.repositoryName;
        cc.filePath = this.filePath;
        cc.resolvedCommitHash = this.resolvedCommitHash;

        this.conflictChunks.add(cc);
    }

    public void addConflictChunk(ConflictChunk cc) {
        this.conflictChunks.add(cc);
    }

    @Override
    public String toString() {
        return "ConflictFile{" +
                "baseContent='" + baseContent + '\'' +
                ", oursContent='" + oursContent + '\'' +
                ", theirsContent='" + theirsContent + '\'' +
                ", mergedContent='" + mergedContent + '\'' +
                ", resolvedContent='" + resolvedContent + '\'' +
                ", resolvedCommitHash='" + resolvedCommitHash + '\'' +
                ", baseHash='" + baseHash + '\'' +
                ", oursHash='" + oursHash + '\'' +
                ", theirsHash='" + theirsHash + '\'' +
                ", filePath='" + filePath + '\'' +
                '}';
    }
}