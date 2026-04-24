package com.ocr.paddleocr.service.Impl;

import com.google.gson.Gson;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.OCRResult;
import com.ocr.paddleocr.domain.Word;
import com.ocr.paddleocr.process.*;
import com.ocr.paddleocr.utils.ImageUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * OCR服务实现类 - 单例模式
 * 负责具体的OCR识别逻辑
 */
@Slf4j
public class OCRServiceImpl {

    private static volatile OCRServiceImpl instance;
    private static volatile OCRServiceImpl customInstance;
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
    private OCRServiceImpl() {
        this(OCRConfig.builder().build());
    }

    /**
     * 私有构造 - 使用自定义配置
     */
    private OCRServiceImpl(OCRConfig ocrConfig) {
        this.ocrConfig = ocrConfig;
        this.gson = new Gson();
        try {
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
            log.info("OCR服务初始化完成");
        } catch (Exception e) {
            log.error("OCR服务实现初始化失败", e);
            throw new RuntimeException("OCR服务实现初始化失败", e);
        }
    }

    /**
     * 获取单例实例（使用默认配置）
     */
    public static OCRServiceImpl getInstance() {
        if (instance == null) {
            synchronized (OCRServiceImpl.class) {
                if (instance == null) {
                    instance = new OCRServiceImpl();
                }
            }
        }
        return instance;
    }

    /**
     * 获取单例实例（使用自定义配置）
     */
    public static OCRServiceImpl getInstance(OCRConfig config) {
        if (customInstance == null) {
            synchronized (OCRServiceImpl.class) {
                if (customInstance == null) {
                    customInstance = new OCRServiceImpl(config);
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
        OCRResult.OCRResultBuilder builder = OCRResult.builder()
                .imagePath(imagePath)
                .success(Boolean.FALSE);

        if (!initialized) {
            return builder.error("OCR服务未初始化").build();
        }
        OCRContext context = new OCRContext();
        long startTime = System.currentTimeMillis();
        try {
            // 读取图片
            Mat rawImage = ImageUtil.getImage(imagePath);
            context.setRawMat(rawImage);
            // 图像检测和切割
            detProcessor.detect(context);
            // 启用分类检测时进行分类检测和纠正
            if (ocrConfig.isUseCls()) {
                clsProcessor.classify(context);
            }
            // 检测框识别
            recProcessor.recognize(context);
            if (ocrConfig.isUseDebug()) {
                DebugProcessor.printBoxes(context,ocrConfig.getDebugPath());
            }
            rawImage.release();
            // 检测结果
            List<Word> words = new ArrayList<>();
            context.getRecResultBoxes().forEach(textBox -> words.add(Word.builder()
                    .text(textBox.getRecText())
                    .confidence(textBox.getRecConfidence())
                    .box(textBox.getRestorePoints())
                    .build()));
            return builder
                    .success(Boolean.TRUE)
                    .words(words)
                    .processingTime(System.currentTimeMillis() - startTime)
                    .build();
        } catch (Exception e) {
            log.error("OCR识别失败: {}", imagePath, e);
            return builder.error(e.getMessage()).build();
        }
    }


    /**
     * 关闭服务，释放资源
     */
    public void shutdown() {
        if (modelManager != null) {
            modelManager.close();
        }
        initialized = false;
        log.info("OCR服务已关闭");
    }
}