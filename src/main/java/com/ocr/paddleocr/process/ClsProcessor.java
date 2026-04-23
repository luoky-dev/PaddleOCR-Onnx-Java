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

    private void preprocess(OCRContext context){
        List<List<TextBox>> clsBatchBoxes = new ArrayList<>();
        List<List<float[]>> clsBatchChw = new ArrayList<>();
        // 按批量处理数量分组
        for (int begin = 0; begin < context.getDetResultBoxes().size(); begin += ocrConfig.getClsBatchSize()) {
            int end = Math.min(begin + ocrConfig.getClsBatchSize(), context.getDetResultBoxes().size());
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

    private void postprocess(OCRContext context){
        int rotatedCount = 0;
        for (int i = 0; i < context.getClsBatchBoxes().size(); i++){
            for (int j = 0; j < context.getClsBatchBoxes().get(i).size(); j++){
                TextBox box = context.getClsBatchBoxes().get(i).get(j);
                float[][] logits = context.getClsLogitsList().get(i);
                int[] decoded = decode(logits[j]);
                int angle = decoded[0];
                float score = Float.intBitsToFloat(decoded[1]);

                box.setAngle(angle);
                box.setClsConfidence(score);
                box.setClsAngle(angle);

                if (needRotate(angle, score)) {
                    rotation(box, angle);
                    if (box.isRotate()) {
                        rotatedCount++;
                    }
                } else {
                    box.setRotate(false);
                }
            }
        }
        context.setClsRotBox(rotatedCount);
    }

    private int[] decode(float[] probs) {
        int bestIdx = 0;
        float best = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > best) {
                best = probs[i];
                bestIdx = i;
            }
        }

        int angle;
        if (probs.length == 2) {
            angle = bestIdx == 1 ? 180 : 0;
        } else if (bestIdx < modelConfig.getFallbackAngleMap().length) {
            angle = modelConfig.getFallbackAngleMap()[bestIdx];
        } else {
            angle = 0;
        }
        return new int[]{angle, Float.floatToIntBits(best)};
    }

    private boolean needRotate(int angle, float score) {
        if (score < ocrConfig.getClsThresh()) {
            return false;
        }
        if (angle == 180) {
            return true;
        }
        return ocrConfig.isUseAngleCls() && (angle == 90 || angle == 270);
    }

    private void rotation(TextBox box, int angle) {
        Mat src = box.getRawMat();
        if (src == null || src.empty()) {
            box.setRotate(false);
            return;
        }
        Mat dst = new Mat();
        if (angle == 180) {
            Core.rotate(src, dst, Core.ROTATE_180);
        } else if (angle == 90) {
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
        } else if (angle == 270) {
            Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
        } else {
            box.setRotate(false);
            dst.release();
            return;
        }
        box.setRotMat(dst);
        box.setRotate(true);
        box.setRotAngle(angle);
    }
}
