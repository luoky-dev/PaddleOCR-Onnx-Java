package com.ocr.paddleocr.domain;

import lombok.Builder;
import lombok.Data;
import org.opencv.core.Point;

import java.io.Serializable;

@Data
@Builder
public class Word implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 文本框四个顶点
     */
    private Point[] box;

    /**
     * 识别的文本内容
     */
    private String text;

    /**
     * 识别置信度 (0-1)
     */
    private float confidence;

}