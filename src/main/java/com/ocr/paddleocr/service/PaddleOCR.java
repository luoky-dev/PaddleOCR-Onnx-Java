package com.ocr.paddleocr.service;

import com.google.gson.Gson;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.*;
import com.ocr.paddleocr.process.ClsProcess;
import com.ocr.paddleocr.process.DetProcess;
import com.ocr.paddleocr.domain.ModelProcessContext;
import com.ocr.paddleocr.process.RecProcess;
import com.ocr.paddleocr.utils.MatPipeline;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * PaddleOCR 服务类
 * 整合检测、方向分类、识别三个模块
 *
 * @author PaddleOCR Team
 */
@Slf4j
public class PaddleOCR implements Closeable {

    // ==================== 核心处理器 ====================

    /** 文本检测处理器 */
    @Getter
    private DetProcess detProcess;

    /** 方向分类处理器 */
    @Getter
    private ClsProcess clsProcess;

    /** 文本识别处理器 */
    @Getter
    private RecProcess recProcess;

    /** OCR配置 */
    @Getter
    private final OCRConfig config;

    /** Gson 序列化器 */
    private final Gson gson;

    /** 初始化状态标志 */
    private volatile boolean initialized = false;

    /** 关闭状态标志 */
    private volatile boolean closed = false;

    /**
     * 使用默认配置创建实例
     */
    public PaddleOCR() throws Exception {
        this(OCRConfig.defaultEnglish());
    }

    /**
     * 使用自定义配置创建实例
     */
    public PaddleOCR(OCRConfig config) throws Exception {
        this.config = config;
        this.gson = new Gson();
        initialize();
        log.info("PaddleOCR 初始化完成，配置: {}", config);
    }

    /**
     * 初始化各个处理器
     */
    private void initialize() throws Exception {
        if (initialized) {
            return;
        }

        long startTime = System.currentTimeMillis();

        config.validate();

        log.info("初始化文本检测处理器...");
        detProcess = new DetProcess(config);

        if (config.isUseAngleCls()) {
            log.info("初始化方向分类处理器...");
            clsProcess = new ClsProcess(config);
        }

        log.info("初始化文本识别处理器...");
        recProcess = new RecProcess(config);

        initialized = true;
        log.info("所有处理器初始化完成，耗时: {}ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 单张图片OCR识别（返回 JSON 字符串）
     *
     * @param imagePath 图片路径
     * @return JSON 格式的识别结果
     */
    public String ocr(String imagePath) {
        return gson.toJson(ocrToObject(imagePath));
    }

    /**
     * 单张图片OCR识别（返回对象）
     *
     * @param imagePath 图片路径
     * @return OCR识别结果对象
     */
    private OCRResult ocrToObject(String imagePath) {
        checkState();

        long startTime = System.currentTimeMillis();
        OCRResult.OCRResultBuilder resultBuilder = OCRResult.builder()
                .imagePath(imagePath)
                .success(false);

        try {
            // 构建模型处理上下文
            ModelProcessContext context = new ModelProcessContext();

            // 1. 读取图像
            Mat image = MatPipeline.fromImage(imagePath).get();

            log.info("开始OCR识别: {}, 图像尺寸: {}x{}", imagePath, image.height(), image.width());

            context.setRawMat(image);
            context.setOriginalHeight(image.height());
            context.setOriginalWidth(image.width());
            resultBuilder.imageHeight(image.height()).imageWidth(image.width());

            // 2. 文本检测
            detProcess.detect(context);
            if (!context.isSuccess() || context.getBoxes() == null || context.getBoxes().isEmpty()) {
                log.warn("未检测到文本区域, 识别失败");
                resultBuilder.error(context.getError())
                        .processingTime(System.currentTimeMillis() - startTime);
                image.release();
                return resultBuilder.build();
            }
            log.info("检测到 {} 个文本区域", context.getBoxes().size());

            // 3. 方向分类
            if (clsProcess != null && config.isUseAngleCls()) {
                clsProcess.classify(context);
                if (!context.isSuccess()) {
                    log.warn("方向分类失败: {}, 将使用原检测结果进行文本识别", context.getError());
                }
                log.info("方向分类完成，处理 {} 个文本区域", context.getClsRotBox());
            }

            // 4. 文本识别
            recProcess.recognize(context);

            if (!context.isSuccess()) {
                log.warn("文本识别失败: {}", context.getError());
                resultBuilder.error(context.getError())
                        .processingTime(System.currentTimeMillis() - startTime);
                image.release();
                return resultBuilder.build();
            }

            // 5. 构建最终结果
            List<OCRPrediction> predictions = convertToPredictions(context.getBoxes());

            resultBuilder.success(true)
                    .predictions(predictions)
                    .allText(buildAllText(predictions))
                    .processingTime(System.currentTimeMillis() - startTime);

            log.info("OCR识别完成，成功识别 {} 个文本区域，总耗时: {}ms",
                    predictions.size(), System.currentTimeMillis() - startTime);

            // 释放图像资源
            image.release();

        } catch (Exception e) {
            log.error("OCR识别失败", e);
            resultBuilder.error(e.getMessage())
                    .processingTime(System.currentTimeMillis() - startTime);
        }
        return resultBuilder.build();
    }

    /**
     * 转换为最终预测结果
     */
    private List<OCRPrediction> convertToPredictions(List<TextBox> boxes) {
        if (boxes == null) {
            return new ArrayList<>();
        }

        return boxes.stream()
                // 过滤空文本和低置信度文本
                .filter(r -> r.getText() != null && !r.getText().isEmpty())
                .filter(r -> r.getRecConfidence() >= config.getMinConfidence())
                .map(r -> OCRPrediction.builder()
                        .box(r.getBox())
                        .text(r.getText())
                        .confidence(r.getRecConfidence())
                        .build())
                .collect(Collectors.toList());
    }

    /**
     * 构建所有识别文本
     */
    private String buildAllText(List<OCRPrediction> predictions) {
        if (predictions == null || predictions.isEmpty()) {
            return "";
        }
        return predictions.stream()
                .map(OCRPrediction::getText)
                .collect(Collectors.joining("\n"));
    }

    /**
     * 检查服务状态
     */
    private void checkState() {
        if (closed) {
            throw new IllegalStateException("PaddleOCR 已经关闭");
        }
        if (!initialized) {
            throw new IllegalStateException("PaddleOCR 未初始化");
        }
    }

    /**
     * 关闭服务，释放资源
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }

        log.info("关闭 PaddleOCR...");

        try {
            if (detProcess != null) {
                detProcess.close();
            }
            if (clsProcess != null) {
                clsProcess.close();
            }
            if (recProcess != null) {
                recProcess.close();
            }
        } catch (Exception e) {
            log.error("关闭资源时出错", e);
        }

        closed = true;
        log.info("PaddleOCR 已关闭");
    }
}