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
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
        long startTime = System.currentTimeMillis();
        // 预处理
        preprocess(context);
        // 模型解析
        parse(context);
        // 后处理
        postprocess(context);
        // 计算运行时间
        context.setClsProcessTime(System.currentTimeMillis() - startTime);
    }

    private void preprocess(OCRContext context) {
        List<List<TextBox>> clsBatchBoxes = new ArrayList<>();
        List<List<float[]>> clsBatchChw = new ArrayList<>();
        // 按批量处理数量分组
        for (int begin = 0; begin < context.getDetResultBoxes().size(); begin += ocrConfig.getBatchSize()) {
            int end = Math.min(begin + ocrConfig.getBatchSize(), context.getDetResultBoxes().size());
            List<TextBox> batch = context.getDetResultBoxes().subList(begin, end);
            // 每一批缩放归一化到chw的模型要求输入格式
            List<float[]> chwList = new ArrayList<>();
            for (TextBox box : batch) {
                Mat src = box.getRestoreMat();
                if (src == null || src.empty()) {
                    chwList.add(new float[3 * modelConfig.getClsModelHeight() * modelConfig.getClsModelWith()]);
                    continue;
                }
                // 缩放归一化
                chwList.add(OpenCVUtil.resizeNormalize(src, modelConfig.getClsModelHeight(), modelConfig.getClsModelWith()));
            }
            clsBatchBoxes.add(batch);
            clsBatchChw.add(chwList);
        }
        context.setClsBatchBoxes(clsBatchBoxes);
        context.setClsBatchChw(clsBatchChw);
    }

    private void parse(OCRContext context) throws OrtException {
        List<float[][]> logitsList = new ArrayList<>();
        for (List<float[]> chwList : context.getClsBatchChw()) {
            // 模型输出
            OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), 3, modelConfig.getClsModelHeight(), modelConfig.getClsModelWith());
            Result output = modelManager.getClsSession().run(Collections.singletonMap("x", input));
            // 模型解析
            float[][] logits = OnnxUtil.parseClsOutput(output);
            logitsList.add(logits);
            output.close();
            input.close();
        }
        context.setClsLogitsList(logitsList);
    }

    private void postprocess(OCRContext context) {
        List<TextBox> clsResultBoxes = new ArrayList<>();
        // 遍历所有批次的检测框
        for (int i = 0; i < context.getClsBatchBoxes().size(); i++){
            // 当前批次的文本框
            List<TextBox> batchBox = context.getClsBatchBoxes().get(i);
            // 当前批次的模型输出
            float[][] logits = context.getClsLogitsList().get(i);
            // 遍历批次内的每个文本框
            for (int j = 0; j < context.getClsBatchBoxes().get(i).size(); j++){
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
                } else {
                    box.setRotate(false);
                }
                clsResultBoxes.add(box);
            }
        }
        context.setClsResultBoxes(clsResultBoxes);
    }

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
            angle = bestIdx == 1 ? 180 : 0;
        } else if (bestIdx < modelConfig.getFallbackAngleMap().length) {
            angle = modelConfig.getFallbackAngleMap()[bestIdx];
        } else {
            angle = 0;
        }
        // 将概率值通过 floatToIntBits 编码为 int，便于存储
        return new int[]{angle, Float.floatToIntBits(best)};
    }

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
        return ocrConfig.isUseCls() && (angle == 90 || angle == 270);
    }

    private void rotation(TextBox box, int angle) {
        // 从 TextBox 获取裁剪后的图像
        Mat src = box.getRestoreMat();
        if (src == null || src.empty()) {
            box.setRotate(false);
            return;
        }
        // 根据角度执行旋转
        Mat dst = new Mat();
        if (angle == 180) {
            // 180度旋转（上下颠倒）
            Core.rotate(src, dst, Core.ROTATE_180);
            box.setRotAngle(180);
        } else if (angle == 90) {
            // 90度顺时针旋转
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
            box.setRotAngle(90);
        } else if (angle == 270) {
            // 90度逆时针旋转（等价于270度顺时针）
            Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
            box.setRotAngle(-90);
        } else {
            box.setRotate(false);
            OpenCVUtil.releaseMat(dst);
            return;
        }
        box.setRotMat(dst);
        box.setRotate(true);
    }
}
