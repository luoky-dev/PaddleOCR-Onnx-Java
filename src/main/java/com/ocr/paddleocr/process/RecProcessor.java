package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.RecBatch;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
public class RecProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;

    public RecProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
    }

    /**
     * 图像识别 - 主流程
     */
    public void recognize(OCRContext context) {
        log.debug("开始图像识别");
        long startTime = System.currentTimeMillis();
        try {
            // 预处理
            preprocess(context);
            // 模型推理
            parse(context);
            // 后处理
            postprocess(context);
        } catch (Exception e) {
            log.error("图像识别失败, 错误信息:",e);
            throw new RuntimeException("Runtime error, recognition failed");
        }
        log.debug("图像识别完成, 识别成功检测框数量: {}, 耗时: {} ms", context.getRecResultBoxes().size(), System.currentTimeMillis() - startTime);
    }

    /**
     * 图像识别 - 预处理
     * 将检测框图像分批转换为模型输入格式
     */
    private void preprocess(OCRContext context) throws OrtException {
        log.debug("图像识别 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // rec模型输入形状和检测框
        List<TextBox> boxes = ocrConfig.isUseCls() ? context.getClsResultBoxes() : context.getDetResultBoxes();
        // 确定模型输入固定高度
        long[] modelInputShape = OnnxUtil.getModelInputShape(modelManager.getRecSession());
        log.debug("图像识别模型输入形状(-1代表动态输入): Batch: {} x Channel: {} x Height:{} x Width:{} ",
                modelInputShape[0], modelInputShape[1], modelInputShape[2], modelInputShape[3]);
        if (modelInputShape[2] == -1 || modelInputShape[3] != -1) {
            log.error("当前暂不支持图像识别模型高度动态或宽度固定输入, 请更换合适的识别模型");
            throw new RuntimeException("Unsupported recognition model, recognition failed");
        }
        int modelInputH = Math.toIntExact(modelInputShape[2]);
        log.debug("模型输入图像尺寸: H:{} x W:{} ", modelInputShape[2], modelInputShape[3]);
        // 透视变换裁剪 + 图像旋转纠正 + 缩放
        Map<TextBox,Mat> originalOrderMap = new HashMap<>();
        // 裁剪图
        Mat rawCropMat = context.getRawMat().clone();
        boxes.forEach(textBox -> {
            // 裁剪
            Mat cropMat = OpenCVUtil.perspectiveTransformCrop(rawCropMat, textBox.getPoints());
            // 将已裁剪区域置空
            OpenCVUtil.fillPolyWhite(rawCropMat, textBox.getPoints());
            log.trace("检测框裁剪完成, 裁剪图尺寸: H:{} x W:{} ", cropMat.height(), cropMat.width());
            // 旋转纠正
            if (textBox.getAngle() != 0 && textBox.isRotate()) {
                OpenCVUtil.rotate(cropMat, textBox.getAngle());
                log.trace("裁剪图按角度旋转纠正完成: {}° -> 0° ", textBox.getAngle());
            }
            // 缩放到固定高度并转换通道
            Size fixHeightSize = OpenCVUtil.getFixHeightSize(cropMat.size(), modelInputH);
            Mat rgbMat = OpenCVUtil.resizeToRGB(cropMat, fixHeightSize);
            log.trace("裁剪图缩放完成: H:{} x W:{} -> H:{} x W:{}",
                    cropMat.height(), cropMat.width(), rgbMat.height(), rgbMat.width());
            originalOrderMap.put(textBox, rgbMat);
            OpenCVUtil.releaseMat(cropMat);
        });
        OpenCVUtil.releaseMat(rawCropMat);
        // 分组并对齐步长倍数 + padding
        int batchSize = ocrConfig.getBatchSize();
        // 转换为List并按宽度排序
        List<Map.Entry<TextBox, Mat>> orderList = new ArrayList<>(originalOrderMap.entrySet());
        orderList.sort(Comparator.comparingInt((Map.Entry<TextBox, Mat> a) -> a.getValue().width()).reversed());
        // 分组
        List<RecBatch> recBatches = new ArrayList<>();
        int batchCount = 0;
        for (int batchBegin = 0; batchBegin < orderList.size(); batchBegin += batchSize) {
            batchCount ++;
            int batchEnd = Math.min(batchBegin + batchSize, orderList.size());
            log.debug("分组预处理第 {} 批开始", batchCount);
            List<TextBox> batchBoxes = new ArrayList<>();
            List<float[]> chwList = new ArrayList<>();
            // 组内最大图像尺寸
            Size maxSize = orderList.get(batchBegin).getValue().size();
            // 最内宽度向上填充到步长的倍数
            Size modelInputSize = OpenCVUtil.widthToStride(maxSize, modelConfig.getStride());
            log.debug("组内最大尺寸: H:{} x W:{}, 组内模型输入尺寸: H:{} x W:{}",
                    maxSize.height, maxSize.width, modelInputSize.height, modelInputSize.width);
            for (int index = batchBegin; index < batchEnd; index++) {
                // 填充
                Mat srcMat = orderList.get(index).getValue();
                Mat paddedMat = OpenCVUtil.padding(srcMat, modelInputSize);
                log.trace("裁剪图填充完成: H:{} x W:{} -> H:{} x W:{}",
                        srcMat.height(), srcMat.width(), paddedMat.height(), paddedMat.width());
                // 归一化并转换CHW格式
                float[] chwData = OpenCVUtil.normalizeToCHW(paddedMat, modelConfig.getLinearMean(), modelConfig.getLinearStd());
                log.trace("裁剪图归一标准化完成, 均值: {}, 标准差: {}",
                        Arrays.toString(modelConfig.getLinearMean()),
                        Arrays.toString(modelConfig.getLinearStd()));
                chwList.add(chwData);
                batchBoxes.add(orderList.get(index).getKey());
                // 资源释放
                OpenCVUtil.releaseMat(srcMat);
                OpenCVUtil.releaseMat(paddedMat);
            }
            log.debug("分组预处理第 {} 批完成, 本批检测框数量: {}, 统一尺寸: H:{} x W:{}",
                    batchCount, batchBoxes.size(), modelInputSize.height, modelInputSize.width);
            recBatches.add(RecBatch.builder()
                    .chwList(chwList)
                    .modelInputSize(modelInputSize)
                    .boxes(batchBoxes)
                    .build());
        }
        context.setRecBatches(recBatches);
        log.debug("图像识别预处理完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 图像识别 - 模型推理
     * 按批次进行模型推理
     */
    private void parse(OCRContext context) throws OrtException {
        log.debug("图像识别 - 模型推理阶段");
        long startTime = System.currentTimeMillis();
        List<RecBatch> recBatch = context.getRecBatches();
        int batchCount = 0;
        for (RecBatch batch : recBatch) {
            batchCount ++;
            // 模型解析输入
            List<float[]> chwList = batch.getChwList();
            // 模型解析
            try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), batch.getModelInputSize());
                 Result output = modelManager.getRecSession().run(Collections.singletonMap("x", input))) {
                // 模型输出
                float[][][] prob = OnnxUtil.parseOnnxValue3D(output);
                batch.setProb(prob);
                log.debug("模型推理第 {}/{} 批完成, 本批检测框数量: {}, 推理字符数: {}, 映射字典数: {}",
                        batchCount, recBatch.size(), prob.length, prob[0].length, prob[0][0].length);
            } catch (OrtException e) {
                log.error("图像识别模型推理失败", e);
                throw e;
            }
        }
        log.debug("模型推理阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);

    }

    /**
     * 图像识别 - 后处理
     * 模型推理结果解码转换为识别文本
     */
    private void postprocess(OCRContext context) {
        log.debug("图像识别 - 后处理解码阶段");
        long startTime = System.currentTimeMillis();
        List<RecBatch> recBatch = context.getRecBatches();
        List<TextBox> recResultBoxes = new ArrayList<>();
        // 读取字典
        String[] dict;
        try {
            dict = OpenCVUtil.readDictionary(ocrConfig.getDictPath());
            log.debug("字典读取成功, 字典长度: {}", dict.length);
        } catch (IOException e) {
            log.error("字典读取失败, 图像识别失败");
            throw new RuntimeException("Read dictionary failed, recognition failed",e);
        }
        // 判断模型输出和字符映射字典是否匹配
        if (recBatch.get(0).getProb()[0][0].length != dict.length) {
            log.error("模型输出与字典类型不匹配, 图像识别失败");
            throw new RuntimeException("String dictionary length is invalid, recognition failed");
        }
        // 按批次解码
        int batchCount = 0;
        int totalCount = 0;
        int filterCount = 0;
        for (RecBatch batch : recBatch) {
            batchCount ++;
            log.debug("当前解码处理第 {}/{} 批次, 本批次检测框数量: {}", batchCount, recBatch.size(), batch.getBoxes().size());
            // 当前批次的模型输出
            float[][][] batchProb = batch.getProb();
            for (int i = 0; i < batch.getBoxes().size(); i++) {
                totalCount ++;
                // 当前检测框框
                TextBox box = batch.getBoxes().get(i);
                // 当前检测框的概率数组
                float[][] boxProb = batchProb[i];
                // 获取所有时间步解码结果
                List<int[]> ctcResult = new ArrayList<>();
                for (float[] timeStep : boxProb) {
                    // 解码: 返回 [最大概率索引, 编码后的概率值]
                    int[] decoded = OpenCVUtil.decode(timeStep);
                    ctcResult.add(decoded);
                }
                log.trace("解码当前批次第 {} 个检测框完成, 置信度阈值: {}", i, ocrConfig.getRecThresh());
                log.trace("当前检测框时间步长度: {}", ctcResult.size());
                log.trace("当前检测框时间步索引: {}",
                        ctcResult.stream().map(arr -> arr[0]).collect(Collectors.toList()));
                log.trace("当前检测框时间步概率: {}",
                        ctcResult.stream().map(arr -> Float.intBitsToFloat(arr[1])).collect(Collectors.toList()));

                // 去除背景和重复时间步
                List<int[]> filteredResult = new ArrayList<>();
                int prevIdx = -1;
                for (int[] arr : ctcResult) {
                    int currentIdx = arr[0];
                    // 遇到background token重置prev
                    if (currentIdx == 0) {
                        prevIdx = -1;
                        continue;
                    }
                    // 只添加与上一个不同的索引
                    if (currentIdx != prevIdx) {
                        filteredResult.add(arr);
                        prevIdx = currentIdx;
                    }
                }

                log.trace("当前检测框时间步过滤后索引: {}",
                        filteredResult.stream().map(arr -> arr[0]).collect(Collectors.toList()));
                log.trace("当前检测框时间步过滤后概率: {}",
                        filteredResult.stream().map(arr -> Float.intBitsToFloat(arr[1])).collect(Collectors.toList()));

                // 提取最终文本和概率
                String recText = filteredResult.stream()
                        .map(arr -> {
                            int idx = arr[0];
                            if (idx >= 0 && idx < dict.length) {
                                return dict[idx];
                            }
                            return "";
                        })
                        .collect(Collectors.joining());

                // 计算平均概率
                double confidence = filteredResult.stream()
                        .mapToDouble(arr -> Float.intBitsToFloat(arr[1]))
                        .average()
                        .orElse(0.0);

                log.trace("当前检测框最终识别结果: {}, 置信度: {}", recText, confidence);
                box.setRecText(recText);
                box.setRecConfidence((float) confidence);
                if(confidence > ocrConfig.getRecThresh()) {
                    recResultBoxes.add(box);
                } else  {
                    filterCount ++;
                    log.trace("当前检测框识别结果置信度过低, 已过滤");
                }
            }
        }
        // 按index阅读顺序重新排列
        recResultBoxes.sort(Comparator.comparing(TextBox::getIndex));
        context.setRecResultBoxes(recResultBoxes);
        log.debug("总解码检测框数量: {}, 低置信度过滤检测框数量: {}, 最终检测框数量: {}", totalCount, filterCount, totalCount - filterCount);
        log.debug("后处理解码阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }
}