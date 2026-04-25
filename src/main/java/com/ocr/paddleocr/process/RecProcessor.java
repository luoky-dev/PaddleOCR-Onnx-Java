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
import java.util.List;

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
        log.debug("RecProcessor初始化完成, 识别模型路径: {}, 批量大小: {}, 图像尺寸: {}x{}, 字典大小: {}",
                ocrConfig.getRecModelPath(),
                ocrConfig.getBatchSize(),
                modelConfig.getRecModelHeight(),
                modelConfig.getRecModelWith(),
                dict.size());
    }

    public void recognize(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        // 检查是否有文本框需要处理
        List<TextBox> sourceBoxes = ocrConfig.isUseCls()
                ? context.getClsResultBoxes()
                : context.getDetResultBoxes();

        if (sourceBoxes == null || sourceBoxes.isEmpty()) {
            log.warn("识别处理跳过：文本框为空");
            context.setRecProcessTime(System.currentTimeMillis() - startTime);
            return;
        }
        // 预处理
        preprocess(context);
        // 模型解析
        parse(context);
        // 后处理
        postprocess(context);
        // 计算运行时间
        context.setRecProcessTime(System.currentTimeMillis() - startTime);
    }

    /**
     * 预处理：将文本框图像分批转换为模型输入格式
     */
    private void preprocess(OCRContext context) {
        long startTime = System.currentTimeMillis();

        List<TextBox> sourceBoxes = ocrConfig.isUseCls()
                ? context.getClsResultBoxes()
                : context.getDetResultBoxes();
        int totalBoxes = sourceBoxes.size();
        int batchSize = ocrConfig.getBatchSize();
        int recHeight = modelConfig.getRecModelHeight();
        int recWidth = modelConfig.getRecModelWith();

        log.debug("识别预处理开始, 总文本框: {}, 批量大小: {}, 目标尺寸: {}x{}",
                totalBoxes, batchSize, recWidth, recHeight);

        List<List<TextBox>> recBatchBoxes = new ArrayList<>();
        List<List<float[]>> recBatchChw = new ArrayList<>();

        int batchCount = 0;
        int processedCount = 0;
        int emptyImageCount = 0;

        for (int begin = 0; begin < totalBoxes; begin += batchSize) {
            int end = Math.min(begin + batchSize, totalBoxes);
            List<TextBox> batch = sourceBoxes.subList(begin, end);
            batchCount++;

            log.debug("处理第 {}/{} 批, 本批文本框数: {}",
                    batchCount, (totalBoxes + batchSize - 1) / batchSize, batch.size());

            // 每一批缩放归一化到chw的模型要求输入格式
            List<float[]> chwList = new ArrayList<>();
            int validCount = 0;

            for (int i = 0; i < batch.size(); i++) {
                TextBox box = batch.get(i);
                // 优先使用旋转校正后的图像，否则使用原始裁剪图
                Mat src = (box.getRotMat() != null && !box.getRotMat().empty())
                        ? box.getRotMat() : box.getRestoreMat();

                if (src == null || src.empty()) {
                    log.warn("第{}批第{}个文本框图像为空，使用零填充", batchCount, i);
                    chwList.add(new float[3 * recHeight * recWidth]);
                    emptyImageCount++;
                    continue;
                }

                // 日志输出图像尺寸（调试用）
                if (log.isDebugEnabled()) {
                    log.debug("文本框[{}] 图像尺寸: {}x{}, 通道数: {}",
                            processedCount + i, src.cols(), src.rows(), src.channels());
                }

                // 缩放归一化
                float[] chwData = OpenCVUtil.resizeNormalize(src, recHeight, recWidth);
                chwList.add(chwData);
                validCount++;
            }

            log.debug("第{}批预处理完成, 有效图像: {}/{}", batchCount, validCount, batch.size());

            recBatchBoxes.add(batch);
            recBatchChw.add(chwList);
            processedCount += batch.size();
        }

        context.setRecBatchBoxes(recBatchBoxes);
        context.setRecBatchChw(recBatchChw);

        long elapsed = System.currentTimeMillis() - startTime;
        log.debug("识别预处理完成, 总批次数: {}, 处理文本框: {}, 空图像: {}, 耗时: {} ms",
                batchCount, processedCount, emptyImageCount, elapsed);
    }

    /**
     * 模型推理：分批执行ONNX推理
     */
    private void parse(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        List<List<float[]>> recBatchChw = context.getRecBatchChw();
        int totalBatches = recBatchChw.size();
        int recHeight = modelConfig.getRecModelHeight();
        int recWidth = modelConfig.getRecModelWith();

        log.debug("识别模型推理开始, 总批次数: {}", totalBatches);

        List<float[][][]> recProbsList = new ArrayList<>();
        int batchIndex = 0;
        int totalSamples = 0;

        for (List<float[]> chwList : recBatchChw) {
            batchIndex++;
            long batchStartTime = System.currentTimeMillis();

            log.debug("执行第 {}/{} 批推理, 本批样本数: {}",
                    batchIndex, totalBatches, chwList.size());

            try (OnnxTensor input = OnnxUtil.createBatchInputTensor(
                    chwList,
                    modelManager.getEnv(),
                    3,
                    recHeight,
                    recWidth
            );
                 OrtSession.Result output = modelManager.getRecSession()
                         .run(Collections.singletonMap("x", input))) {

                // 模型解析
                float[][][] probs = OnnxUtil.parseRecOutput(output);
                recProbsList.add(probs);
                totalSamples += probs.length;

                long batchElapsed = System.currentTimeMillis() - batchStartTime;
                log.debug("第{}批推理完成, 耗时: {}ms, 输出形状: {}x{}x{}",
                        batchIndex, batchElapsed, probs.length,
                        probs[0].length, probs[0][0].length);

            } catch (OrtException e) {
                log.error("第{}批推理失败", batchIndex, e);
                throw e;
            }
        }

        context.setRecProbsList(recProbsList);

        long elapsed = System.currentTimeMillis() - startTime;
        log.info("识别模型推理完成, 总批次数: {}, 总样本数: {}, 耗时: {}ms",
                totalBatches, totalSamples, elapsed);
    }


    /**
     * 后处理：CTC解码，将模型输出转换为文本
     */
    private void postprocess(OCRContext context) {
        long startTime = System.currentTimeMillis();
        log.debug("识别后处理开始, 总批次数: {}", context.getRecBatchBoxes().size());
        List<TextBox> recResultBoxes = new ArrayList<>();
        // 遍历所有批次的检测框
        for (int i = 0; i < context.getRecBatchBoxes().size(); i++) {
            // 当前批次的文本框
            List<TextBox> batchBox = context.getRecBatchBoxes().get(i);
            // 当前批次的模型输出
            float[][][] probs = context.getRecProbsList().get(i);
            log.debug("处理第{}批后处理, 文本框数: {}, 输出数: {}",
                    i + 1, batchBox.size(), probs.length);
            // 遍历批次内的每个文本框
            for (int j = 0; j < batchBox.size(); j++) {
                // 当前文本框
                TextBox box = batchBox.get(j);
                ctcDecode(box, probs[j]);
                recResultBoxes.add(box);
            }
        }
        context.setRecResultBoxes(recResultBoxes);
        long elapsed = System.currentTimeMillis() - startTime;
        log.info("识别后处理完成, 耗时: {} ms, 总文本框: {}",
                elapsed, recResultBoxes.size());
    }

    /**
     * CTC解码：将模型输出的概率序列转换为文本
     *
     * @param box 文本框对象（会设置识别结果和置信度）
     * @param timeSteps 模型输出 [time_steps, num_classes]
     */
    private void ctcDecode(TextBox box, float[][] timeSteps) {
        if (timeSteps == null || timeSteps.length == 0) {
            log.warn("CTC解码失败: 时间步序列为空");
            box.setRecText("");
            box.setRecConfidence(0);
            return;
        }
        int numClasses = timeSteps[0].length;
        int dictSize = dict.size();

        // 计算索引偏移量
        int offset;
        if (numClasses == dictSize + 1) {
            offset = -1;
            log.debug("检测到包含blank token, numClasses={}, dictSize={}, offset={}",
                    numClasses, dictSize, offset);
        } else if (numClasses == dictSize) {
            offset = 0;
            log.debug("检测到不包含blank token, numClasses={}, dictSize={}, offset={}",
                    numClasses, dictSize, offset);
        } else if (numClasses > dictSize + 1) {
            offset = -1;
            log.debug("numClasses({}) > dictSize+1({}), 使用offset={}",
                    numClasses, dictSize + 1, offset);
        } else {
            offset = 0;
            log.warn("numClasses({}) < dictSize({}), 使用offset={}",
                    numClasses, dictSize, offset);
        }

        StringBuilder sb = new StringBuilder();
        float confSum = 0.0f;
        int confCount = 0;
        int prev = -1;

        for (float[] step : timeSteps) {
            int bestIdx = 0;
            float bestProb = step[0];
            // 找出当前时间步的最大概率索引
            for (int i = 1; i < step.length; i++) {
                if (step[i] > bestProb) {
                    bestProb = step[i];
                    bestIdx = i;
                }
            }
            // 跳过重复的连续字符（CTC collapse）
            if (bestIdx == prev) {
                continue;
            }
            prev = bestIdx;
            // 跳过blank token
            if (offset == -1 && bestIdx == modelConfig.getBlankIndex()) {
                continue;
            }
            // 转换为字典索引, 字典索引越界跳过
            int dictIdx = bestIdx + offset;
            if (dictIdx < 0 || dictIdx >= dict.size()) {
                continue;
            }
            // 获取字符并添加到结果
            String token = dict.get(dictIdx);
            sb.append(token);
            confSum += bestProb;
            confCount++;
        }
        String result = sb.toString().trim();
        float confidence = confCount > 0 ? (confSum / confCount) : 0.0f;
        box.setRecText(result);
        box.setRecConfidence(confidence);
        if (log.isDebugEnabled() && !result.isEmpty()) {
            log.debug("CTC解码完成: 文本='{}', 置信度={}, 时间步={}",
                    result, String.format("%.3f", confidence),
                    timeSteps.length);
        }
    }

}
