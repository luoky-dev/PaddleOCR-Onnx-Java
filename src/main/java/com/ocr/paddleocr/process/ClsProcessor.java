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
        preprocess(context);
        // 模型推理
        parse(context);
        // 后处理
        postprocess(context);
        log.info("分类检测完成, 角度分类完成检测框数量: {}, 耗时: {} ms", context.getClsResultBoxes().size(), System.currentTimeMillis() - startTime);
    }

    /**
     * 预处理: 将检测框图像分批转换为模型输入格式
     */
    private void preprocess(OCRContext context) throws OrtException {
        log.info("分类检测 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // cls模型输入形状和检测框
        List<TextBox> boxes = context.getDetResultBoxes();
        long[] modelInputShape = OnnxUtil.getModelInputShape(modelManager.getClsSession());
        log.debug("方向分类模型输入形状(-1代表动态输入): Batch: {} x Channel: {} x Height:{} x Width:{} ",
                modelInputShape[0], modelInputShape[1], modelInputShape[2], modelInputShape[3]);
        // 总数和批量大小
        int batchSize = ocrConfig.getBatchSize();
        log.debug("检测框数量: {}, 批量处理大小: {}, 总批次: {}", boxes.size(), batchSize, (boxes.size() + batchSize - 1) / batchSize);
        // 确定模型输入尺寸
        // cls模型对图像文字形变不敏感, 可以直接简单分组缩放输入
        int modelInputH = Math.toIntExact(modelInputShape[2] != -1 ? modelInputShape[2] : modelConfig.getClsModelHeight());
        int modelInputW = Math.toIntExact(modelInputShape[3] != -1 ? modelInputShape[3] : modelConfig.getClsModelWith());
        log.debug("模型输入图像尺寸: H:{} x W:{} ", modelInputH, modelInputW);
        // 按batch简单分组
        List<ClsBatch> clsBatches = new ArrayList<>();
        int batchCount = 0;
        for (int batchBegin = 0; batchBegin < boxes.size(); batchBegin += batchSize) {
            batchCount ++;
            // 分组
            int batchEnd = Math.min(batchBegin + batchSize, boxes.size());
            log.debug("分组预处理第 {} 批开始", batchCount);
            List<TextBox> batchBoxes = boxes.subList(batchBegin, batchEnd);
            // 当前批次检测框直接缩放归一到模型输入尺寸
            List<float[]> chwList = new ArrayList<>();
            // 裁剪图
            Mat rawCropMat = context.getRawMat().clone();
            for (TextBox textBox : batchBoxes) {
                // 透视变换裁剪
                Mat cropMat = OpenCVUtil.perspectiveTransformCrop(rawCropMat, textBox.getPoints());
                // 将已裁剪区域置空
                OpenCVUtil.fillPolyWhite(rawCropMat, textBox.getPoints());
                log.trace("图像裁剪完成, 裁剪图尺寸: H:{} x W:{} ", cropMat.height(), cropMat.width());
                // 缩放和转换转换RGB通道
                Mat rgbMat = OpenCVUtil.resizeToRGB(cropMat,new Size(modelInputW, modelInputH));
                log.trace("图像缩放完成: H:{} x W:{} -> H:{} x W:{}",
                        cropMat.height(), cropMat.width(), rgbMat.height(), rgbMat.width());
                // 归一化并转换CHW格式
                float[] chwData = OpenCVUtil.normalizeToCHW(rgbMat, modelConfig.getLinearMean(), modelConfig.getLinearStd());
                log.trace("图像归一标准化完成, 均值: {}, 标准差: {}",
                        Arrays.toString(modelConfig.getLinearMean()),
                        Arrays.toString(modelConfig.getLinearStd()));
                chwList.add(chwData);
                // 资源释放
                OpenCVUtil.releaseMat(cropMat);
                OpenCVUtil.releaseMat(rgbMat);
            }
            OpenCVUtil.releaseMat(rawCropMat);
            log.debug("分组预处理第 {} 批完成, 本批检测框数量: {}, 统一尺寸: H:{} x W:{}",
                    batchCount, batchBoxes.size(), modelInputH, modelInputW);
            clsBatches.add(ClsBatch.builder()
                    .chwList(chwList)
                    .modelInputSize(new Size(modelInputW, modelInputH))
                    .boxes(batchBoxes)
                    .build());
        }
        context.setClsBatches(clsBatches);
        log.info("分类检测预处理完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 模型推理：分批执行ONNX推理
     */
    private void parse(OCRContext context) throws OrtException {
        log.info("分类检测 - 模型推理阶段");
        long startTime = System.currentTimeMillis();
        List<ClsBatch> clsBatch = context.getClsBatches();
        int batchCount = 0;
        for (ClsBatch batch : clsBatch) {
            batchCount ++;
            // 模型解析输入
            List<float[]> chwList = batch.getChwList();
            // 模型解析
            try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), batch.getModelInputSize());
                 Result output = modelManager.getClsSession().run(Collections.singletonMap("x", input))) {
                // 模型输出
                float[][] prob = OnnxUtil.parseClsOutput(output);
                batch.setProb(prob);
                log.debug("模型推理第 {}/{} 批完成, 本批检测框数量: {}, 方向类别数量: {}", batchCount, clsBatch.size(), prob.length, prob[0].length);
            } catch (OrtException e) {
                log.error("方向分类模型推理失败", e);
                throw e;
            }
        }
        log.info("模型推理阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 后处理: 解码输出并执行旋转
     */
    private void postprocess(OCRContext context) {
        log.info("分类检测 - 后处理检测框旋转纠正阶段");
        long startTime = System.currentTimeMillis();
        List<ClsBatch> clsBatch = context.getClsBatches();
        List<TextBox> clsResultBoxes = new ArrayList<>();
        // 判断模型输出和角度分类字典是否匹配
        if (clsBatch.get(0).getProb()[0].length != modelConfig.getAngleDict().length) {
            log.error("模型输出与字典类型不匹配, 分类检测后处理失败");
            throw new RuntimeException("Angle dictionary length is invalid, angle classify decoding failed");
        }
        // 按批次解码
        int batchCount = 0;
        for (ClsBatch batch : clsBatch) {
            batchCount ++;
            log.debug("当前解码处理第 {}/{} 批次, 本批次检测框数量: {}", batchCount, clsBatch.size(), batch.getBoxes().size());
            // 当前批次的模型输出
            float[][] prob = batch.getProb();
            for (int i = 0; i < batch.getBoxes().size(); i++) {
                // 当前检测框框
                TextBox box = batch.getBoxes().get(i);
                // 当前检测框框方向分类概率数组
                int[] decoded = OpenCVUtil.decode(prob[i]);
                // 角度
                int angle = modelConfig.getAngleDict()[decoded[0]];
                // 置信度
                float score = Float.intBitsToFloat(decoded[1]);
                log.trace("解码当前批次第 {} 个检测框完成, 角度: {}, 置信度: {}, 正常阈值: {}", i, angle, score, ocrConfig.getClsThresh());
                // 设值
                box.setAngle(angle);
                box.setClsConfidence(score);
                // 旋转纠正图像
                if (score > ocrConfig.getClsThresh() && angle != 0) {
                    box.setRotate(true);
                    log.trace("当前检测框角度非正向角度且置信度超过阈值, 需要执行旋转纠正操作");
                } else {
                    box.setRotate(false);
                    log.trace("当前检测框角度正常或非正向角度置信度过低");
                }
                clsResultBoxes.add(box);
            }
        }
        context.setClsResultBoxes(clsResultBoxes);

        // 统计按角度分组的旋转数量
        log.debug("检测框角度分类统计: 总检测框数量: {}, 非正向检测框数量: {}, 角度统计: 90°: {}, 180°: {}, 270°: {}",
                clsResultBoxes.size(),
                clsResultBoxes.stream().filter(TextBox::isRotate).count(),
                clsResultBoxes.stream().filter(box -> box.isRotate() && box.getAngle() == 90).count(),
                clsResultBoxes.stream().filter(box -> box.isRotate() && box.getAngle() == 180).count(),
                clsResultBoxes.stream().filter(box -> box.isRotate() && box.getAngle() == 270).count());
        log.info("后处理检测框旋转纠正阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }
}