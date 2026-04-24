package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.ImageUtil;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import org.opencv.core.Mat;

import java.util.*;

public class RecProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;
    private final List<String> dict;
    private final Set<Integer> ignoredTokens;

    public RecProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
        this.dict = ImageUtil.readDictionary(ocrConfig.getDictPath());
        this.ignoredTokens = initIgnoredTokens(dict);
    }

    public void recognize(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        // 预处理
        preprocess(context);
        // 模型解析
        parse(context);
        // 后处理
        postprocess(context);
        // 计算运行时间
        context.setRecProcessTime(System.currentTimeMillis() - startTime);
    }

    private void preprocess(OCRContext context) {
        List<List<TextBox>> recBatchBoxes = new ArrayList<>();
        List<List<float[]>> recBatchChw = new ArrayList<>();
        List<TextBox> resultBoxes;
        if(ocrConfig.isUseCls()) {
            resultBoxes = context.getClsResultBoxes();
        } else {
            resultBoxes = context.getDetResultBoxes();
        }
        // 按批量处理数量分组
        for (int begin = 0; begin < resultBoxes.size(); begin += ocrConfig.getBatchSize()) {
            int end = Math.min(begin + ocrConfig.getBatchSize(), resultBoxes.size());
            List<TextBox> batch = resultBoxes.subList(begin, end);
            // 每一批缩放归一化到chw的模型要求输入格式
            List<float[]> chwList = new ArrayList<>();
            for (TextBox box : batch) {
                Mat src = (box.getRotMat() != null && !box.getRotMat().empty())
                        ? box.getRotMat() : box.getRestoreMat();
                if (src == null || src.empty()) {
                    chwList.add(new float[3 * modelConfig.getRecModelHeight() * modelConfig.getRecModelWith()]);
                    continue;
                }
                // 缩放归一化
                chwList.add(OpenCVUtil.resizeNormalize(src, modelConfig.getRecModelHeight(), modelConfig.getRecModelWith()));
            }
            recBatchBoxes.add(batch);
            recBatchChw.add(chwList);
        }
        context.setRecBatchBoxes(recBatchBoxes);
        context.setRecBatchChw(recBatchChw);
    }

    private void parse(OCRContext context) throws OrtException {
        List<float[][][]> recProbsList = new ArrayList<>();
        for (List<float[]> chwList : context.getRecBatchChw()) {
            // 模型输出
            OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), 3, modelConfig.getRecModelHeight(), modelConfig.getRecModelWith());
            OrtSession.Result output = modelManager.getRecSession().run(Collections.singletonMap("x", input));
            // 模型解析
            float[][][] probs = OnnxUtil.parseRecOutput(output);
            recProbsList.add(probs);
            output.close();
            input.close();
        }
        context.setRecProbsList(recProbsList);
    }

    private void postprocess(OCRContext context) {
        List<TextBox> recResultBoxes = new ArrayList<>();
        // 遍历所有批次的检测框
        for (int i = 0; i < context.getRecBatchBoxes().size(); i++) {
            // 当前批次的文本框
            List<TextBox> batchBox = context.getRecBatchBoxes().get(i);
            // 当前批次的模型输出
            float[][][] probs = context.getRecProbsList().get(i);
            // 遍历批次内的每个文本框
            for (int j = 0; j < context.getRecBatchBoxes().get(i).size(); j++) {
                // 当前文本框
                TextBox box = batchBox.get(j);
                ctcDecode(box, probs[j]);
                recResultBoxes.add(box);
            }
        }
        context.setRecResultBoxes(recResultBoxes);
    }

    private void ctcDecode(TextBox box, float[][] timeSteps) {
        if (timeSteps == null || timeSteps.length == 0) {
            box.setRecText("");
            box.setRecConfidence(0);
            return;
        }

        StringBuilder sb = new StringBuilder();
        float confSum = 0.0f;
        int confCount = 0;
        int prev = modelConfig.getBlankIndex();

        // 官方思路：按时间步做 greedy 取最大，再做 CTC 去重与去 blank。
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

            if (ignoredTokens.contains(bestIdx)) {
                continue;
            }

            int dictIdx = bestIdx - 1; // 0 is blank
            if (dictIdx >= 0 && dictIdx < dict.size()) {
                sb.append(dict.get(dictIdx));
                confSum += bestProb;
                confCount++;
            }
        }

        String text = sb.toString().trim();
        float conf = confCount > 0 ? (confSum / confCount) : 0.0f;
        box.setRecText(text);
        box.setRecConfidence(conf);
    }

    private Set<Integer> initIgnoredTokens(List<String> dictionary) {
        // 创建忽略 token 集合
        Set<Integer> set = new HashSet<>();
        // CTC 解码中的空白符，表示无输出
        set.add(modelConfig.getBlankIndex());
        // 判断是否需要保留空格, 只有英文等西方语言需要保留空格
        boolean useSpaceChar =
                !"ch".equals(ocrConfig.getLang()) &&
                        !"chi".equals(ocrConfig.getLang()) &&
                        !"japan".equals(ocrConfig.getLang()) &&
                        !"korean".equals(ocrConfig.getLang());
        // 对于不使用空格的语言，忽略空格 token
        if (!useSpaceChar) {
            // 在字典中查找空格字符的位置
            int spacePos = -1;
            for (int i = 0; i < dictionary.size(); i++) {
                if (" ".equals(dictionary.get(i))) {
                    // 索引 +1 因为 blank token 占据了索引 0
                    spacePos = i + 1;
                    break;
                }
            }
            // 如果找到空格，将其加入忽略集合
            if (spacePos > 0) {
                set.add(spacePos);
            }
        }
        return set;
    }
}
