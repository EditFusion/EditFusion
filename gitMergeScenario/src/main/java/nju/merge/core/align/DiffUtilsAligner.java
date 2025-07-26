package nju.merge.core.align;

import difflib.DiffUtils;
import difflib.Patch;
import nju.merge.entity.ConflictChunk;
import nju.merge.entity.ConflictFile;

import java.util.*;

public class DiffUtilsAligner extends Aligner {

    public static void getResolutions(ConflictFile cf) {
        List<String> merged = Arrays.stream(cf.mergedContent).toList();
        List<String> truth = Arrays.stream(cf.resolvedContent).toList();
        List<String> mergedWithoutChunks = new ArrayList<>();
        Map<Integer, Integer> chunkIdx2line = new HashMap<>();

        int idx = 0;
        for (int i = 0; i < merged.size();) {
            if (!merged.get(i).startsWith("<<<<<<<")) {
                mergedWithoutChunks.add(merged.get(i));
                i++;
                continue;
            }

            // chunk met
            assert idx < cf.conflictChunks.size();
            assert  cf.conflictChunks.get(idx).startLine == i;
            chunkIdx2line.put(idx++, mergedWithoutChunks.size());
            while (!merged.get(i).startsWith(">>>>>>>")) {
                i++;
            }
            i++;
        }

        int[] match = alignLines(mergedWithoutChunks, truth);
        for (int i = 0; i < cf.conflictChunks.size(); i++) {
            assert chunkIdx2line.containsKey(i);
            ConflictChunk cc = cf.conflictChunks.get(i);
            int lineAfterRemoving = chunkIdx2line.get(i);

            int truthStart, truthEnd;
            if (lineAfterRemoving == 0) {
                truthStart = 0;
            } else {
                if (match[lineAfterRemoving - 1] == -1) continue;
                truthStart = match[lineAfterRemoving - 1] + 1;
            }
            if (lineAfterRemoving == mergedWithoutChunks.size()) {
                truthEnd = truth.size();
            } else {
                if (match[lineAfterRemoving] == -1) continue;
                truthEnd = match[lineAfterRemoving];
            }

            cc.resolution = truth.subList(truthStart, truthEnd).toArray(new String[0]);
        }
    }

    private static int[] alignLines(List<String> mergedWithoutChunks, List<String> truth) {
        int n = mergedWithoutChunks.size();
        int m = truth.size();
        int[] match = new int[n];
        Arrays.fill(match, -1);

        Patch<String> patch = DiffUtils.diff(mergedWithoutChunks, truth);
        List<String> diff = DiffUtils.generateUnifiedDiff("", "", mergedWithoutChunks, patch, Math.max(n, m));

        for(int i = 0, j = 0, k = 3; k < diff.size(); ++k){
            char c = diff.get(k).charAt(0);
            if(c == '-')
                i++;
            else if(c == '+')
                j++;
            else{
                match[i] = j;
                i++;
                j++;
            }
        }
        return match;
    }
}