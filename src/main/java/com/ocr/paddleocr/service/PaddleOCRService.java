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
    private volatile PaddleOCRServiceImpl ocrService;
    private volatile OCRConfig config;

    /**
     * 私有构造
     */
    private PaddleOCRService(OCRConfig config) {
        this.config = config;
        this.ocrService = new PaddleOCRServiceImpl(config);
        log.debug("OCR服务初始化完成");
    }

    /**
     * 获取实例 - 支持配置更新
     */
    private static PaddleOCRService getInstance(OCRConfig config) {
        if (instance == null) {
            synchronized (PaddleOCRService.class) {
                if (instance == null) {
                    instance = new PaddleOCRService(config);
                } else {
                    // 检查配置是否变更
                    if (!instance.config.equals(config)) {
                        log.info("检测到配置变更, 重新初始化OCR服务");
                        instance.reinit(config);
                    }
                }
            }
        }
        return instance;
    }

    /**
     * 重新初始化服务
     */
    private synchronized void reinit(OCRConfig newConfig) {
        try {
            // 关闭旧服务
            if (ocrService != null) {
                ocrService.shutdown();
            }
            // 创建新服务
            this.ocrService = new PaddleOCRServiceImpl(newConfig);
            this.config = newConfig;
            log.info("OCR服务重新初始化完成");
        } catch (Exception e) {
            log.error("OCR服务重新初始化失败", e);
            throw new RuntimeException("Failed to reinitialize OCR service", e);
        }
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