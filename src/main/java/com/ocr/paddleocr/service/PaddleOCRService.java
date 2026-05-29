package com.ocr.paddleocr.service;

import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.service.Impl.PaddleOCRServiceImpl;
import lombok.extern.slf4j.Slf4j;

/**
 * OCR服务 - 支持静态方法调用
 * 提供两种静态方法：
 * 1. recognize(imagePath) - 使用默认配置
 * 2. recognize(config, imagePath) - 使用自定义配置
 */
@Slf4j
public class PaddleOCRService {

    private static volatile PaddleOCRService instance;
    private static volatile PaddleOCRService customInstance;

    private final PaddleOCRServiceImpl ocrService;

    /**
     * 私有构造 - 使用默认配置
     */
    private PaddleOCRService() {
        this.ocrService = PaddleOCRServiceImpl.getInstance();
        log.info("OCR服务初始化完成（默认配置）");
    }

    /**
     * 私有构造 - 使用自定义配置
     */
    private PaddleOCRService(OCRConfig config) {
        this.ocrService = PaddleOCRServiceImpl.getInstance(config);
        log.info("OCR服务初始化完成（自定义配置）");
    }

    /**
     * 获取单例实例（使用默认配置）
     */
    public static PaddleOCRService getInstance() {
        if (instance == null) {
            synchronized (PaddleOCRService.class) {
                if (instance == null) {
                    instance = new PaddleOCRService();
                }
            }
        }
        return instance;
    }

    /**
     * 获取单例实例（使用自定义配置）
     */
    public static PaddleOCRService getInstance(OCRConfig config) {
        if (customInstance == null) {
            synchronized (PaddleOCRService.class) {
                if (customInstance == null) {
                    customInstance = new PaddleOCRService(config);
                } else {
                    log.warn("使用自定义配置初始化, 新配置将被忽略");
                }
            }
        }
        return customInstance;
    }

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
     * 重启服务
     */
    public void restart() {
        ocrService.restart();
    }
    /**
     * 关闭服务
     */
    public void shutdown() {
        ocrService.shutdown();
    }
}