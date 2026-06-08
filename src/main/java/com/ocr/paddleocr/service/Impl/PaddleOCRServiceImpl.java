package com.ocr.paddleocr.service.Impl;

import com.google.gson.Gson;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.OCRResult;
import com.ocr.paddleocr.domain.Word;
import com.ocr.paddleocr.process.*;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * OCR服务实现类 - 单例模式
 * 负责具体的OCR识别逻辑
 */
@Slf4j
public class PaddleOCRServiceImpl {

    private static volatile PaddleOCRServiceImpl instance;
    private static volatile PaddleOCRServiceImpl customInstance;
    private final Gson gson;
    private final ModelManager modelManager;
    private final DetProcessor detProcessor;
    private final ClsProcessor clsProcessor;
    private final RecProcessor recProcessor;
    private final OCRConfig ocrConfig;
    private volatile boolean initialized;

    /**
     * 私有构造 - 使用默认配置
     */
    private PaddleOCRServiceImpl() {
        this(OCRConfig.builder().build());
    }

    /**
     * 私有构造 - 使用自定义配置
     */
    private PaddleOCRServiceImpl(OCRConfig ocrConfig) {
        if (ocrConfig == null) {
            log.error("OCR服务配置不能为空");
            throw new IllegalArgumentException("OCR service configuration cannot be empty");
        }
        this.gson = new Gson();
        try {
            ocrConfig.validate();
            this.ocrConfig = ocrConfig;
            this.modelManager = ModelManager.getInstance();
            synchronized (modelManager) {
                if (!modelManager.isInitialized()) {
                    modelManager.init(ocrConfig);
                }
            }
            this.detProcessor = new DetProcessor(modelManager);
            this.clsProcessor = new ClsProcessor(modelManager);
            this.recProcessor = new RecProcessor(modelManager);
            this.initialized = true;
            log.debug("OCR服务初始化完成");
        } catch (Exception e) {
            log.error("OCR服务实现初始化失败", e);
            throw new RuntimeException("OCR service initialization failed", e);
        }
    }

    /**
     * 获取单例实例 - 使用默认配置
     */
    public static PaddleOCRServiceImpl getInstance() {
        if (instance == null) {
            synchronized (PaddleOCRServiceImpl.class) {
                if (instance == null) {
                    instance = new PaddleOCRServiceImpl();
                }
            }
        }
        return instance;
    }

    /**
     * 获取单例实例 - 使用自定义配置
     */
    public static PaddleOCRServiceImpl getInstance(OCRConfig config) {
        if (customInstance == null) {
            synchronized (PaddleOCRServiceImpl.class) {
                if (customInstance == null) {
                    customInstance = new PaddleOCRServiceImpl(config);
                } else {
                    log.warn("OCRServiceImpl已使用自定义配置初始化, 新配置将被忽略");
                }
            }
        }
        return customInstance;
    }

    /**
     * 识别图片并返回JSON字符串
     *
     * @param imagePath 图片路径
     * @return JSON格式的识别结果
     */
    public String recognize(String imagePath) {
        OCRResult result = rec(imagePath);
        return gson.toJson(result);
    }

    /**
     * 识别图片并返回OCRResult对象
     *
     * @param imagePath 图片路径
     * @return OCRResult对象
     */
    private OCRResult rec(String imagePath) {
        log.debug("PaddleOCR服务开始");
        long startTime = System.currentTimeMillis();
        OCRResult.OCRResultBuilder builder = OCRResult.builder()
                .imagePath(imagePath)
                .success(Boolean.FALSE);
        OCRContext context = new OCRContext();
        context.setImagePath(imagePath);
        try {
            if (!initialized) {
                log.error("OCR服务未初始化, 识别失败");
                throw new RuntimeException("OCR service not initialized, recognition failed");
            }
            // 读取图片
            context.setRawMat(OpenCVUtil.getImage(imagePath));
            context.setImageName(OpenCVUtil.getImageName(imagePath));
            log.debug("图片读取成功, 图片路径: {}, 图片名: {}", context.getImagePath(), context.getImageName());
            // 图像检测
            detProcessor.detect(context);
            if (context.getDetResultBoxes().isEmpty()){
                log.error("未检测到文本框, 识别失败");
                throw new RuntimeException("No text box detected, recognition failed");
            }
            // 启用分类检测时进行分类检测和纠正
            if (ocrConfig.isUseCls()) {
                log.debug("方向分类检测已启用");
                clsProcessor.classify(context);
            } else {
                log.debug("方向分类检测未启用, 将跳过方向分类使用检测模型结果进行识别");
            }
            // 检测框识别
            recProcessor.recognize(context);
            if (context.getRecResultBoxes().isEmpty()){
                log.error("无文本框识别结果, 识别失败");
                throw new RuntimeException("No text box recognition result, recognition failed");
            }
            // debug打印中间图像信息
            if (ocrConfig.isUseDebug()) {
                log.debug("Debug模式已启用, 打印中间图像信息到 {} 目录", ocrConfig.getDebugPath());
                DebugProcessor.printDebugImages(context, ocrConfig);
            }
            List<Word> words = new ArrayList<>();
            StringBuilder allTextStr = new StringBuilder();
            context.getRecResultBoxes().forEach(textBox -> {
                words.add(Word.builder()
                        .text(textBox.getRecText())
                        .confidence(textBox.getRecConfidence())
                        .box(textBox.getPoints())
                        .build());
                allTextStr.append(textBox.getRecText()).append(" ");
            });

            return builder
                    .success(Boolean.TRUE)
                    .imageWidth(context.getRawMat().width())
                    .imageHeight(context.getRawMat().height())
                    .allText(allTextStr.toString())
                    .words(words)
                    .processingTime(System.currentTimeMillis() - startTime)
                    .build();
        } catch (Exception e) {
            log.error("OCR识别失败, 错误信息: ", e);
            return builder
                    .success(Boolean.FALSE)
                    .error(e.getMessage())
                    .build();
        } finally {
            OpenCVUtil.releaseMat(context.getRawMat());
        }
    }

    public synchronized void restart() {
        shutdown();
        try {
            synchronized (modelManager) {
                if (!modelManager.isInitialized()) {
                    modelManager.init(ocrConfig);
                }
            }
            initialized = true;
            log.debug("OCR服务已重启");
        } catch (Exception e) {
            log.error("OCR服务重启失败", e);
            throw new RuntimeException("OCR service failed to restart", e);
        }
    }

    public void shutdown() {
        if (modelManager != null) {
            modelManager.close();
        }
        initialized = false;
        log.debug("OCR服务已关闭");
    }
}
