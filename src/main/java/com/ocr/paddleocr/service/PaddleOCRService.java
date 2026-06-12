package com.ocr.paddleocr.service;

import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.service.Impl.PaddleOCRServiceImpl;
import lombok.extern.slf4j.Slf4j;

/**
 * OCR服务 - 支持静态方法调用
 */
@Slf4j
public class PaddleOCRService {

    private static volatile PaddleOCRService instance;
    private final PaddleOCRServiceImpl ocrService;

    /**
     * 私有构造
     */
    private PaddleOCRService(OCRConfig config) {
        this.ocrService = PaddleOCRServiceImpl.getInstance(config);
        log.debug("OCR服务初始化完成（自定义配置）");
    }

    /**
     * 获取单例实例
     */
    public static PaddleOCRService getInstance(OCRConfig config) {
        if (instance == null) {
            synchronized (PaddleOCRService.class) {
                if (instance == null) {
                    instance = new PaddleOCRService(config);
                } else {
                    log.warn("使用自定义配置初始化, 新配置将被忽略");
                }
            }
        }
        return instance;
    }

    /**
     * 静态方法 - 识别图片
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