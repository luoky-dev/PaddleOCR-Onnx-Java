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
        List<ClsBatch> clsBatch = preprocess(context);
        // 模型解析
        parse(clsBatch);
        // 后处理
        List<TextBox> clsResultBoxes = postprocess(clsBatch);
        // 设置结果
        long elapsed = System.currentTimeMillis() - startTime;
        context.setClsResultBoxes(clsResultBoxes);
        context.setClsProcessTime(elapsed);
        log.info("分类检测完成, 耗时: {} ms", elapsed);
    }

    /**
     * 预处理: 将检测框图像分批转换为模型输入格式
     */
    private List<ClsBatch> preprocess(OCRContext context) throws OrtException {
        log.info("分类检测 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // cls模型输入形状和检测框
        List<TextBox> detBoxes = context.getDetResultBoxes();
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
                // 透视变换裁剪
                Mat cropMat = OpenCVUtil.perspectiveTransformCrop(context.getRawMat(), textBox.getRestorePoints());
                // 缩放和转换转换RGB通道
                Mat rgbMat = OpenCVUtil.resizeToRGB(cropMat,new Size(modelInputW, modelInputH));
                log.debug("图像缩放完成: H:{} x W:{} -> H:{} x W:{}",
                        cropMat.height(), cropMat.width(), rgbMat.height(), rgbMat.width());
                // 归一化并转换CHW格式
                float[] chwData = OpenCVUtil.normalizeToCHW(rgbMat, modelConfig.getLinearMean(), modelConfig.getLinearStd());
                log.debug("图像归一标准化完成, 均值: {}, 标准差: {}",
                        Arrays.toString(modelConfig.getLinearMean()),
                        Arrays.toString(modelConfig.getLinearStd()));
                chwList.add(chwData);
                // 资源释放
                OpenCVUtil.releaseMat(cropMat);
                OpenCVUtil.releaseMat(rgbMat);
            }
            clsBatches.add(ClsBatch.builder().
                    chwList(chwList).
                    modelInputSize(new Size(modelInputW, modelInputH)).
                    boxes(batchBoxes)
                    .build());
        }
        log.info("分类检测预处理完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
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
                log.debug("模型推理第 {}/{} 批完成, 本批检测框数量: {}, 方向类别数量: {}", batchCount, clsBatches.size(), probVector.length, probVector[0].length);
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
    private List<TextBox> postprocess(List<ClsBatch> clsBatch) {
        log.info("分类检测 - 后处理检测框旋转纠正阶段");
        long startTime = System.currentTimeMillis();
        // 判断模型输出和角度分类字典是否匹配
        if (clsBatch.get(0).getProbVector()[0].length != modelConfig.getAngleDict().length) {
            log.error("模型输出与字典类型不匹配, 分类检测后处理失败");
            throw new RuntimeException("Angle dictionary length is invalid, angle classify decoding failed");
        }
        // 按批次解码
        int batchCount = 0;
        List<TextBox> clsResultBoxes = new ArrayList<>();
        for (ClsBatch batch : clsBatch) {
            batchCount ++;
            log.debug("当前解码处理第 {}/{} 批次, 本批次检测框数量: {}", batchCount, clsBatch.size(), batch.getBoxes().size());
            // 当前批次的模型输出
            float[][] probVector = batch.getProbVector();
            for (int i = 0; i < batch.getBoxes().size(); i++) {
                // 当前检测框框
                TextBox box = batch.getBoxes().get(i);
                // 当前检测框框方向分类概率数组
                String[] decoded = OpenCVUtil.decode(probVector[i], modelConfig.getAngleDict());
                // 角度
                int angle = Integer.parseInt(decoded[2]);
                // 置信度
                float score = Float.parseFloat(decoded[1]);
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

        // 统计按角度分组的旋转数量
        long rotate180Count = clsResultBoxes.stream()
                .filter(box -> box.isRotate() && box.getAngle() == 180)
                .count();
        long rotate90Count = clsResultBoxes.stream()
                .filter(box -> box.isRotate() && box.getAngle() == 90)
                .count();
        long rotate270Count = clsResultBoxes.stream()
                .filter(box -> box.isRotate() && box.getAngle() == 270)
                .count();
        long rotatedCount = rotate180Count + rotate90Count + rotate270Count;

        log.debug("检测框旋转纠正统计: 总检测框数量: {}, 触发旋转纠正检测框数量: {} , 角度统计: 180°: {}, 90°: {}, 270°: {}",
                clsResultBoxes.size(), rotatedCount, rotate180Count, rotate90Count, rotate270Count);
        log.info("后处理检测框旋转纠正阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
        return clsResultBoxes;
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

}