package com.ocr.paddleocr.domain;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;
import java.util.List;

/**
 * OCR识别结果
 *
 */
@Data
@Builder
public class OCRResult implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 原始图片路径
     */
    private String imagePath;

    /**
     * 图片宽度
     */
    private int imageWidth;

    /**
     * 图片高度
     */
    private int imageHeight;

    /**
     * 处理时间（毫秒）
     */
    private long processingTime;

    /**
     * 是否成功
     */
    private boolean success;

    /**
     * 错误信息
     */
    private String error;

    /**
     * 所有识别文本
     */
    private String allText;

    /**
     * 预测结果列表
     */
    private List<Word> words;
}