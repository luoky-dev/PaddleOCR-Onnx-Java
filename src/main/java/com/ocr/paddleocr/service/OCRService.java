package com.ocr.paddleocr.service;

import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.service.Impl.OCRServiceImpl;
import lombok.extern.slf4j.Slf4j;

/**
 * OCR服务 - 支持静态方法调用
 * 提供两种静态方法：
 * 1. recognize(imagePath) - 使用默认配置
 * 2. recognize(config, imagePath) - 使用自定义配置
 */
@Slf4j
public class OCRService {

    private static volatile OCRService instance;
    private static volatile OCRService customInstance;

    private final OCRServiceImpl ocrService;

    /**
     * 私有构造 - 使用默认配置
     */
    private OCRService() {
        this.ocrService = OCRServiceImpl.getInstance();
        log.info("OCR服务初始化完成（默认配置）");
    }

    /**
     * 私有构造 - 使用自定义配置
     */
    private OCRService(OCRConfig config) {
        this.ocrService = OCRServiceImpl.getInstance(config);
        log.info("OCR服务初始化完成（自定义配置）");
    }

    /**
     * 获取单例实例（使用默认配置）
     */
    public static OCRService getInstance() {
        if (instance == null) {
            synchronized (OCRService.class) {
                if (instance == null) {
                    instance = new OCRService();
                }
            }
        }
        return instance;
    }

    /**
     * 获取单例实例（使用自定义配置）
     */
    public static OCRService getInstance(OCRConfig config) {
        if (customInstance == null) {
            synchronized (OCRService.class) {
                if (customInstance == null) {
                    customInstance = new OCRService(config);
                } else {
                    log.warn("使用自定义配置初始化, 新配置将被忽略");
                }
            }
        }
        return customInstance;
    }

    // ==================== 静态方法（供开发者调用） ====================

    /**
     * 静态方法：识别图片（使用默认配置）
     *
     * @param imagePath 图片路径
     * @return JSON格式的识别结果
     */
    public static String recognize(String imagePath) {
        return getInstance().ocrService.recognize(imagePath);
    }

    /**
     * 静态方法：识别图片（使用自定义配置）
     *
     * @param config OCR配置
     * @param imagePath 图片路径
     * @return JSON格式的识别结果
     */
    public static String recognize(OCRConfig config, String imagePath) {
        return getInstance(config).ocrService.recognize(imagePath);
    }

    /**
     * 关闭服务
     */
    public void shutdown() {
        ocrService.shutdown();
    }
}