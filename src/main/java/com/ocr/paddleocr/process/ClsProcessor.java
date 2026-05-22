package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ClsBatch;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.util.*;

@Slf4j
public class ClsProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;

    public ClsProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
    }

    public void classify(OCRContext context) throws OrtException {
        log.info("开始分类检测");
        long startTime = System.currentTimeMillis();
        // 预处理
        List<ClsBatch> clsState = preprocess(context.getDetResultBoxes());
        // 模型解析
        parse(clsState);
        // 后处理
        postprocess(context);
        // 计算运行时间
        context.setClsProcessTime(System.currentTimeMillis() - startTime);
    }

    /**
     * 预处理: 将检测框图像分批转换为模型输入格式
     */
    private List<ClsBatch> preprocess(List<TextBox> detBoxes) throws OrtException {
        log.info("分类检测 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // cls模型输入形状和检测框
        long[] modelInputShape = OnnxUtil.getModelInputShape(modelManager.getClsSession());
        log.debug("方向分类模型输入形状(-1代表动态输入): Batch: {} x Channel: {} x Height:{} x Width:{} ",
                modelInputShape[0], modelInputShape[1], modelInputShape[2], modelInputShape[3]);
        // 总数和批量大小
        int batchSize = ocrConfig.getBatchSize();
        log.debug("检测框数量: {}, 批量处理大小: {}, 总批次: {}", detBoxes.size(), batchSize, (detBoxes.size() + batchSize - 1) / batchSize);
        // 确定模型输入尺寸
        int modelInputH = Math.toIntExact(modelInputShape[2] != -1 ? modelInputShape[2] : modelConfig.getClsModelHeight());
        int modelInputW = Math.toIntExact(modelInputShape[3] != -1 ? modelInputShape[3] : modelConfig.getClsModelWith());
        log.debug("模型输入图像尺寸: H:{} x W:{} ", modelInputH, modelInputW);
        // 按batch简单分组
        List<ClsBatch> clsBatches = new ArrayList<>();
        int batchCount = 0;
        for (int batchBegin = 0; batchBegin < detBoxes.size(); batchBegin += batchSize) {
            batchCount ++;
            // 分组
            int batchEnd = Math.min(batchBegin + batchSize, detBoxes.size());
            List<TextBox> batchBoxes = detBoxes.subList(batchBegin, batchEnd);
            log.debug("分组预处理第 {} 批, 本批检测框数量: {}", batchCount, batchBoxes.size());
            // 当前批次检测框直接缩放归一到模型输入尺寸
            List<float[]> chwList = new ArrayList<>();
            for (TextBox textBox : batchBoxes) {
                // 缩放和转换转换RGB通道
                Mat rgbMat = OpenCVUtil.resizeToRGB(textBox.getCropMat(),new Size(modelInputW, modelInputH));
                log.debug("图像缩放完成: H:{} x W:{} -> H:{} x W:{}",
                        textBox.getCropMat().height(), textBox.getCropMat().width(), rgbMat.height(), rgbMat.width());
                // 归一化并转换CHW格式
                float[] chwData = OpenCVUtil.normalizeToCHW(rgbMat, modelConfig.getLinearMean(), modelConfig.getLinearStd());
                log.debug("图像归一标准化完成, 均值: {}, 标准差: {}",
                        Arrays.toString(modelConfig.getLinearMean()),
                        Arrays.toString(modelConfig.getLinearStd()));
                chwList.add(chwData);
                // 资源释放
                OpenCVUtil.releaseMat(rgbMat);
            }
            clsBatches.add(ClsBatch.builder().
                    chwList(chwList).
                    modelInputSize(new Size(modelInputW, modelInputH)).
                    boxes(batchBoxes)
                    .build());
        }
        log.debug("分类检测预处理完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
        return clsBatches;
    }

    /**
     * 模型推理：分批执行ONNX推理
     */
    private void parse(List<ClsBatch> clsBatches) throws OrtException {
        log.info("分类检测 - 模型推理阶段");
        long startTime = System.currentTimeMillis();
        int batchCount = 0;
        for (ClsBatch clsBatch : clsBatches) {
            batchCount ++;
            // 模型解析输入
            List<float[]> chwList = clsBatch.getChwList();
            // 模型解析
            try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), clsBatch.getModelInputSize());
                 Result output = modelManager.getClsSession().run(Collections.singletonMap("x", input))) {
                // 模型输出
                float[][] probVector = OnnxUtil.parseClsOutput(output);
                clsBatch.setProbVector(probVector);
                log.info("模型推理第 {} 批完成, 本批检测框数量: {}, 方向类别数量: {}", batchCount, probVector.length, probVector[0].length);
            } catch (OrtException e) {
                log.error("方向分类模型推理失败", e);
                throw e;
            }
        }
        log.info("模型推理阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 后处理：解码输出并执行旋转
     */
    private void postprocess(OCRContext context) {
        log.info("分类检测 - 后处理检测框旋转纠正阶段");
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

        log.info("分类后处理完成, 耗时: {} ms, 总检测框: {}, 旋转: {} (180°: {}, 90°: {}, 270°: {})",
                elapsed, totalBoxes, rotatedCount, rotate180Count, rotate90Count, rotate270Count);
    }

    /**
     * 按检测框高度聚类
     * @param boxes 检测框列表
     * @param strideSize 分组间隔
     * @return 按高度分组的Map, Key为高度区间起始值
     */
    private Map<Integer,List<TextBox>> heightGroup(List<TextBox> boxes,int strideSize) {
        Map<Integer, List<TextBox>> heightGroups = new HashMap<>();
        for (TextBox box : boxes) {
            // 检测框的高度
            int height = box.getCropMat().height();
            // 计算分组key, 向上取整到strideSize的倍数
            int groupKey = ((height + strideSize - 1) / strideSize) * strideSize;
            // 将检测框添加到对应分组
            heightGroups.computeIfAbsent(groupKey, k -> new ArrayList<>()).add(box);
        }
        return heightGroups;
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
        Mat src = box.getCropMat();
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