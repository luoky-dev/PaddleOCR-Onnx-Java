package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

@Slf4j
public class RecProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;
    private final List<String> dict;

    public RecProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
        this.dict = OpenCVUtil.readDictionary(ocrConfig.getDictPath());
        log.debug("RecProcessor initialized, modelPath: {}, batchSize: {}, recInputHeight: {}, dictSize: {}",
                ocrConfig.getRecModelPath(),
                ocrConfig.getBatchSize(),
                modelConfig.getRecModelHeight(),
                dict.size());
    }

    public void recognize(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        List<TextBox> sourceBoxes = ocrConfig.isUseCls()
                ? context.getClsResultBoxes()
                : context.getDetResultBoxes();

        if (sourceBoxes == null || sourceBoxes.isEmpty()) {
            log.warn("Recognition skipped because text boxes are empty");
            context.setRecProcessTime(System.currentTimeMillis() - startTime);
            return;
        }

        RecState state = preprocess(sourceBoxes);
        parse(state);
        List<TextBox> recResultBoxes = postprocess(state);

        context.setRecResultBoxes(recResultBoxes);
        context.setRecProcessTime(System.currentTimeMillis() - startTime);
        log.info("Recognition done, totalBoxes: {}, elapsed: {} ms",
                recResultBoxes.size(), context.getRecProcessTime());
    }

    private RecState preprocess(List<TextBox> sourceBoxes) throws OrtException {
        long startTime = System.currentTimeMillis();
        boolean dynamicWidth = OnnxUtil.isDynamicInput(modelManager.getRecSession());
        int batchSize = ocrConfig.getBatchSize();
        int recHeight = modelConfig.getRecModelHeight();
        int fixedRecWidth = modelConfig.getRecModelWith();

        List<TextBox> originalOrder = new ArrayList<>(sourceBoxes);
        List<TextBox> sortedBoxes = new ArrayList<>(sourceBoxes);
        sortedBoxes.sort(Comparator.comparingDouble(this::getAspectRatio));

        List<RecBatch> batches = new ArrayList<>();
        for (int begin = 0; begin < sortedBoxes.size(); begin += batchSize) {
            int end = Math.min(begin + batchSize, sortedBoxes.size());
            List<TextBox> batchBoxes = new ArrayList<>(sortedBoxes.subList(begin, end));
            int batchWidth = resolveBatchWidth(batchBoxes, recHeight, fixedRecWidth, dynamicWidth);
            List<float[]> batchChw = preprocessBatch(batchBoxes, recHeight, batchWidth);
            batches.add(new RecBatch(batchBoxes, batchChw, batchWidth));
        }

        long elapsed = System.currentTimeMillis() - startTime;
        log.info("Recognition preprocess done, totalBoxes: {}, totalBatches: {}, dynamicWidth: {}, elapsed: {} ms",
                sourceBoxes.size(), batches.size(), dynamicWidth, elapsed);

        return new RecState(originalOrder, batches, recHeight);
    }

    private List<float[]> preprocessBatch(List<TextBox> batchBoxes, int recHeight, int batchWidth) {
        List<float[]> batchChw = new ArrayList<>(batchBoxes.size());
        for (TextBox box : batchBoxes) {
            Mat src = getSourceMat(box);
            if (src == null || src.empty()) {
                batchChw.add(new float[3 * recHeight * batchWidth]);
                continue;
            }
            batchChw.add(OpenCVUtil.resizeNormalize(src, recHeight, batchWidth));
        }
        return batchChw;
    }

    private void parse(RecState state) throws OrtException {
        long startTime = System.currentTimeMillis();
        int batchIndex = 0;
        int totalSamples = 0;

        for (RecBatch batch : state.batches) {
            batchIndex++;
            long batchStart = System.currentTimeMillis();

            batch.probs = runRecBatchWithRetry(batch.chwList, state.recHeight, batch.batchWidth, 0);
            totalSamples += batch.probs.length;

            long batchElapsed = System.currentTimeMillis() - batchStart;
            log.info("Recognition parse batch {}/{} done, batchSize: {}, batchWidth: {}, elapsed: {} ms",
                    batchIndex, state.batches.size(), batch.boxes.size(), batch.batchWidth, batchElapsed);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        log.info("Recognition parse done, totalBatches: {}, totalSamples: {}, elapsed: {} ms",
                state.batches.size(), totalSamples, elapsed);
    }

    private List<TextBox> postprocess(RecState state) {
        long startTime = System.currentTimeMillis();
        List<TextBox> decodedBoxes = new ArrayList<>(state.originalOrder.size());

        for (RecBatch batch : state.batches) {
            for (int i = 0; i < batch.boxes.size(); i++) {
                TextBox box = batch.boxes.get(i);
                ctcDecode(box, batch.probs[i]);
                decodedBoxes.add(box);
            }
        }

        Map<TextBox, Integer> orderMap = new IdentityHashMap<>();
        for (int i = 0; i < state.originalOrder.size(); i++) {
            orderMap.put(state.originalOrder.get(i), i);
        }
        decodedBoxes.sort(Comparator.comparingInt(box -> orderMap.getOrDefault(box, Integer.MAX_VALUE)));

        long elapsed = System.currentTimeMillis() - startTime;
        log.info("Recognition postprocess done, totalBoxes: {}, elapsed: {} ms",
                decodedBoxes.size(), elapsed);
        return decodedBoxes;
    }

    private int resolveBatchWidth(List<TextBox> batchBoxes,
                                  int recHeight,
                                  int fixedRecWidth,
                                  boolean dynamicWidth) {
        if (!dynamicWidth) {
            return fixedRecWidth;
        }

        double maxRatio = 1.0d;
        for (TextBox box : batchBoxes) {
            Mat src = getSourceMat(box);
            if (src == null || src.empty()) {
                continue;
            }
            maxRatio = Math.max(maxRatio, (double) src.cols() / Math.max(1, src.rows()));
        }

        int rawWidth = (int) Math.ceil(recHeight * maxRatio);
        return alignWidth(Math.max(modelConfig.getResizeAlign(), rawWidth), modelConfig.getResizeAlign());
    }

    private int alignWidth(int width, int align) {
        if (align <= 1) {
            return width;
        }
        return ((width + align - 1) / align) * align;
    }

    private double getAspectRatio(TextBox box) {
        Mat src = getSourceMat(box);
        if (src == null || src.empty()) {
            return Double.MAX_VALUE;
        }
        return (double) src.cols() / Math.max(1, src.rows());
    }

    private Mat getSourceMat(TextBox box) {
        if (box == null) {
            return null;
        }
        if (box.getRotMat() != null && !box.getRotMat().empty()) {
            return box.getRotMat();
        }
        return box.getRestoreMat();
    }

    private float[][][] runRecBatchWithRetry(List<float[]> chwList,
                                             int recHeight,
                                             int recWidth,
                                             int splitDepth) throws OrtException {
        try (OnnxTensor input = OnnxUtil.createBatchInputTensor(
                chwList,
                modelManager.getEnv(),
                3,
                recHeight,
                recWidth
        );
             OrtSession.Result output = modelManager.getRecSession()
                     .run(Collections.singletonMap("x", input))) {
            return OnnxUtil.parseRecOutput(output);
        } catch (OrtException e) {
            if (!isCudaOutOfMemory(e) || chwList.size() <= 1) {
                throw e;
            }

            int mid = chwList.size() / 2;
            List<float[]> left = new ArrayList<>(chwList.subList(0, mid));
            List<float[]> right = new ArrayList<>(chwList.subList(mid, chwList.size()));

            log.warn("Recognition batch OOM, split and retry: batchSize={} -> {} + {}, depth={}",
                    chwList.size(), left.size(), right.size(), splitDepth + 1);

            float[][][] leftProbs = runRecBatchWithRetry(left, recHeight, recWidth, splitDepth + 1);
            float[][][] rightProbs = runRecBatchWithRetry(right, recHeight, recWidth, splitDepth + 1);
            return mergeRecProbs(leftProbs, rightProbs);
        }
    }

    private boolean isCudaOutOfMemory(OrtException e) {
        if (e == null || e.getMessage() == null) {
            return false;
        }
        String message = e.getMessage().toLowerCase();
        return message.contains("out of memory")
                || message.contains("cuda failure 2")
                || message.contains("cudnn_status_alloc_failed");
    }

    private float[][][] mergeRecProbs(float[][][] left, float[][][] right) {
        float[][][] merged = new float[left.length + right.length][][];
        System.arraycopy(left, 0, merged, 0, left.length);
        System.arraycopy(right, 0, merged, left.length, right.length);
        return merged;
    }

    private void ctcDecode(TextBox box, float[][] timeSteps) {
        if (timeSteps == null || timeSteps.length == 0) {
            log.warn("Recognition decode failed because timeSteps is empty");
            box.setRecText("");
            box.setRecConfidence(0.0f);
            return;
        }

        int numClasses = timeSteps[0].length;
        int dictSize = dict.size();
        int offset;
        if (numClasses == dictSize + 1) {
            offset = -1;
        } else if (numClasses == dictSize) {
            offset = 0;
        } else if (numClasses > dictSize + 1) {
            offset = -1;
        } else {
            offset = 0;
            log.warn("Recognition output classes {} smaller than dict size {}", numClasses, dictSize);
        }

        StringBuilder sb = new StringBuilder();
        float confSum = 0.0f;
        int confCount = 0;
        int prev = -1;

        for (float[] step : timeSteps) {
            int bestIdx = 0;
            float bestProb = step[0];
            for (int i = 1; i < step.length; i++) {
                if (step[i] > bestProb) {
                    bestProb = step[i];
                    bestIdx = i;
                }
            }

            if (bestIdx == prev) {
                continue;
            }
            prev = bestIdx;

            if (offset == -1 && bestIdx == modelConfig.getBlankIndex()) {
                continue;
            }

            int dictIdx = bestIdx + offset;
            if (dictIdx < 0 || dictIdx >= dict.size()) {
                continue;
            }

            sb.append(dict.get(dictIdx));
            confSum += bestProb;
            confCount++;
        }

        box.setRecText(sb.toString().trim());
        box.setRecConfidence(confCount > 0 ? (confSum / confCount) : 0.0f);
    }

    private static final class RecState {
        private final List<TextBox> originalOrder;
        private final List<RecBatch> batches;
        private final int recHeight;

        private RecState(List<TextBox> originalOrder, List<RecBatch> batches, int recHeight) {
            this.originalOrder = originalOrder;
            this.batches = batches;
            this.recHeight = recHeight;
        }
    }

    private static final class RecBatch {
        private final List<TextBox> boxes;
        private final List<float[]> chwList;
        private final int batchWidth;
        private float[][][] probs;

        private RecBatch(List<TextBox> boxes, List<float[]> chwList, int batchWidth) {
            this.boxes = boxes;
            this.chwList = chwList;
            this.batchWidth = batchWidth;
        }
    }
}
