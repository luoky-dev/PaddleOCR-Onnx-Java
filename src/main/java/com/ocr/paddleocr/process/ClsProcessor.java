package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Slf4j
public class ClsProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;

    public ClsProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
        log.debug("ClsProcessor初始化完成, 分类模型路径: {}, 批量大小: {}, 方向分类模型输入高宽: {}x{}",
                ocrConfig.getClsModelPath(),
                ocrConfig.getBatchSize(),
                modelConfig.getClsModelHeight(),
                modelConfig.getClsModelWith());
    }

    public void classify(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        // 检查是否有文本框需要处理
        if (context.getDetResultBoxes() == null || context.getDetResultBoxes().isEmpty()) {
            log.warn("分类处理跳过：检测结果为空");
            context.setClsProcessTime(System.currentTimeMillis() - startTime);
            return;
        }
        // 预处理
        preprocess(context);
        // 模型解析
        parse(context);
        // 后处理
        postprocess(context);
        // 计算运行时间
        context.setClsProcessTime(System.currentTimeMillis() - startTime);
    }

    /**
     * 预处理：将检测框图像分批转换为模型输入格式
     */
    private void preprocess(OCRContext context) {
        long startTime = System.currentTimeMillis();
        List<TextBox> detBoxes = context.getDetResultBoxes();
        int batchSize = ocrConfig.getBatchSize();
        int totalBoxes = detBoxes.size();
        int actualBatchSize = Math.min(batchSize, totalBoxes);

        log.debug("分类预处理开始, 总文本框: {}, 配置批量大小: {}, 实际批量大小: {}",
                totalBoxes, batchSize, actualBatchSize);

        List<List<TextBox>> clsBatchBoxes = new ArrayList<>();
        List<List<float[]>> clsBatchChw = new ArrayList<>();

        int batchCount = 0;
        int processedCount = 0;

        // 按批量处理数量分组
        for (int begin = 0; begin < totalBoxes; begin += batchSize) {
            int end = Math.min(begin + batchSize, totalBoxes);
            List<TextBox> batch = detBoxes.subList(begin, end);
            batchCount++;

            log.debug("处理第 {}/{} 批, 本批文本框数: {}", batchCount,
                    (totalBoxes + batchSize - 1) / batchSize, batch.size());

            // 每一批缩放归一化到chw的模型要求输入格式
            List<float[]> chwList = new ArrayList<>();
            int validCount = 0;

            for (int i = 0; i < batch.size(); i++) {
                TextBox box = batch.get(i);
                Mat src = box.getRestoreMat();

                if (src == null || src.empty()) {
                    log.warn("第{}批第{}个文本框图像为空，使用零填充", batchCount, i);
                    chwList.add(new float[3 * modelConfig.getClsModelHeight() * modelConfig.getClsModelWith()]);
                    continue;
                }

                // 日志输出图像尺寸（调试用）
                if (log.isDebugEnabled()) {
                    log.debug("文本框图像尺寸: {}x{}, 通道数: {}",
                            src.cols(), src.rows(), src.channels());
                }

                // 缩放归一化
                float[] chwData = OpenCVUtil.resizeNormalize(src,
                        modelConfig.getClsModelHeight(),
                        modelConfig.getClsModelWith());
                chwList.add(chwData);
                validCount++;
            }

            log.debug("第{}批预处理完成, 有效图像: {}/{}", batchCount, validCount, batch.size());

            clsBatchBoxes.add(batch);
            clsBatchChw.add(chwList);
            processedCount += batch.size();
        }

        context.setClsBatchBoxes(clsBatchBoxes);
        context.setClsBatchChw(clsBatchChw);

        long elapsed = System.currentTimeMillis() - startTime;
        log.debug("分类预处理完成, 总批次数: {}, 处理文本框: {}, 耗时: {} ms",
                batchCount, processedCount, elapsed);
    }

    /**
     * 模型推理：分批执行ONNX推理
     */
    private void parse(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        List<List<float[]>> clsBatchChw = context.getClsBatchChw();
        int totalBatches = clsBatchChw.size();

        log.debug("分类模型推理开始, 总批次数: {}", totalBatches);

        List<float[][]> logitsList = new ArrayList<>();
        int batchIndex = 0;

        for (List<float[]> chwList : clsBatchChw) {
            batchIndex++;
            long batchStartTime = System.currentTimeMillis();

            log.debug("执行第 {}/{} 批推理, 本批样本数: {}", batchIndex, totalBatches, chwList.size());

            // 模型输出
            try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList,
                    modelManager.getEnv(),
                    3,
                    modelConfig.getClsModelHeight(),
                    modelConfig.getClsModelWith());
                 Result output = modelManager.getClsSession().run(Collections.singletonMap("x", input))) {

                // 模型解析
                float[][] logits = OnnxUtil.parseClsOutput(output);
                logitsList.add(logits);

                long batchElapsed = System.currentTimeMillis() - batchStartTime;
                log.debug("第{}批推理完成, 耗时: {} ms, 输出形状: {}x{}",
                        batchIndex, batchElapsed, logits.length, logits[0].length);
            } catch (OrtException e) {
                log.error("第{}批推理失败", batchIndex, e);
                throw e;
            }
        }

        context.setClsLogitsList(logitsList);

        long elapsed = System.currentTimeMillis() - startTime;
        log.info("分类模型推理完成, 总批次数: {}, 总样本数: {}, 耗时: {} ms",
                totalBatches,
                logitsList.stream().mapToInt(arr -> arr.length).sum(),
                elapsed);
    }

    /**
     * 后处理：解码输出并执行旋转
     */
    private void postprocess(OCRContext context) {
        long startTime = System.currentTimeMillis();
        List<List<TextBox>> clsBatchBoxes = context.getClsBatchBoxes();
        List<float[][]> logitsList = context.getClsLogitsList();

        int totalBatches = clsBatchBoxes.size();
        log.debug("分类后处理开始, 总批次数: {}", totalBatches);

        List<TextBox> clsResultBoxes = new ArrayList<>();
        int rotatedCount = 0;
        int totalBoxes = 0;

        // 遍历所有批次的检测框
        for (int i = 0; i < totalBatches; i++) {
            // 当前批次的文本框
            List<TextBox> batchBox = clsBatchBoxes.get(i);
            // 当前批次的模型输出
            float[][] logits = logitsList.get(i);

            log.debug("处理第{}批后处理, 文本框数: {}, 输出数: {}", i + 1, batchBox.size(), logits.length);

            // 遍历批次内的每个文本框
            for (int j = 0; j < batchBox.size(); j++) {
                // 当前文本框
                TextBox box = batchBox.get(j);
                // 解码模型输出
                int[] decoded = decode(logits[j]);
                // 预测的角度（0°, 90°, 180°, 270°）
                int angle = decoded[0];
                // 预测置信度
                float score = Float.intBitsToFloat(decoded[1]);

                box.setAngle(angle);
                box.setClsConfidence(score);
                // 判断角度和旋转纠正
                if (needRotate(angle, score)) {
                    rotation(box, angle);
                    if (box.isRotate()) {
                        rotatedCount++;
                    }
                } else {
                    box.setRotate(false);
                }
                clsResultBoxes.add(box);
            }
            totalBoxes += batchBox.size();
        }

        context.setClsResultBoxes(clsResultBoxes);

        long elapsed = System.currentTimeMillis() - startTime;

        // 统计按角度分组的旋转数量
        long rotate180Count = clsResultBoxes.stream()
                .filter(box -> box.isRotate() && box.getRotAngle() == 180)
                .count();
        long rotate90Count = clsResultBoxes.stream()
                .filter(box -> box.isRotate() && box.getRotAngle() == 90)
                .count();
        long rotate270Count = clsResultBoxes.stream()
                .filter(box -> box.isRotate() && box.getRotAngle() == -90)
                .count();

        log.info("分类后处理完成, 耗时: {} ms, 总文本框: {}, 旋转: {} (180°: {}, 90°: {}, 270°: {})",
                elapsed, totalBoxes, rotatedCount, rotate180Count, rotate90Count, rotate270Count);
    }

    /**
     * 解码模型输出，获取角度和置信度
     */
    private int[] decode(float[] probs) {
        // 找出最大概率的索引
        int bestIdx = 0;
        float best = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > best) {
                best = probs[i];
                bestIdx = i;
            }
        }
        // 根据输出维度映射角度
        int angle;
        if (probs.length == 2) {
            // 二分类：[0°, 180°]
            angle = bestIdx == 1 ? 180 : 0;
        } else if (bestIdx < modelConfig.getFallbackAngleMap().length) {
            // 多分类：使用映射表
            angle = modelConfig.getFallbackAngleMap()[bestIdx];
        } else {
            log.warn("未知的分类索引: {}, 使用默认角度0", bestIdx);
            angle = 0;
        }
        // 将概率值通过 floatToIntBits 编码为 int，便于存储
        return new int[]{angle, Float.floatToIntBits(best)};
    }

    /**
     * 判断是否需要旋转
     */
    private boolean needRotate(int angle, float score) {
        // 置信度不足，不旋转
        if (score < ocrConfig.getClsThresh()) {
            return false;
        }
        // 180度必须旋转
        if (angle == 180) {
            return true;
        }
        // 90/270度可选旋转
        boolean needRotate = ocrConfig.isUseCls() && (angle == 90 || angle == 270);
        if (needRotate && log.isDebugEnabled()) {
            log.debug("检测到{}度旋转, 启用旋转校正", angle);
        }
        return needRotate;
    }

    /**
     * 执行图像旋转
     */
    private void rotation(TextBox box, int angle) {
        // 从 TextBox 获取裁剪后的图像
        Mat src = box.getRestoreMat();
        if (src == null || src.empty()) {
            log.warn("旋转失败: 文本框图像为空, angle={}", angle);
            box.setRotate(false);
            return;
        }

        // 根据角度执行旋转
        Mat dst = new Mat();
        String rotateType;

        if (angle == 180) {
            // 180度旋转（上下颠倒）
            Core.rotate(src, dst, Core.ROTATE_180);
            box.setRotAngle(180);
            rotateType = "180°";
        } else if (angle == 90) {
            // 90度顺时针旋转
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
            box.setRotAngle(90);
            rotateType = "90°顺时针";
        } else if (angle == 270) {
            // 90度逆时针旋转（等价于270度顺时针）
            Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
            box.setRotAngle(-90);
            rotateType = "90°逆时针(270°)";
        } else {
            log.warn("不支持的旋转角度: {}, 跳过旋转", angle);
            box.setRotate(false);
            OpenCVUtil.releaseMat(dst);
            return;
        }

        box.setRotMat(dst);
        box.setRotate(true);

        if (log.isDebugEnabled()) {
        log.debug("图像旋转完成: {}旋转, 原图尺寸: {}x{}, 旋转后尺寸: {}x{}",
                rotateType, src.cols(), src.rows(), dst.cols(), dst.rows());
        }
    }
}