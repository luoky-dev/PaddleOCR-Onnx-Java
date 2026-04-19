package com.ocr.paddleocr.domain;

import com.ocr.paddleocr.domain.TextBox;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.opencv.core.Mat;

import java.util.List;

/**
 * 模型处理上下文
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ModelProcessContext {

    /**
     * 处理前的图像数据
     */
    private Mat rawMat;

    /**
     * 检测预处理后的图像数据
     */
    private Mat detPrepMat;

    /**
     * 原始图像宽度
     */
    private int originalWidth;

    /**
     * 原始图像高度
     */
    private int originalHeight;

    /**
     * 检测预处理后图像宽度
     */
    private int detPrepWidth;

    /**
     * 检测预处理后图像高度
     */
    private int detPrepHeight;

    /**
     * 缩放比例
     */
    private float scale;

    /**
     * 模型输出
     */
    private float[][] modelOutput;

    /**
     * 处理时间（毫秒）
     */
    private long detProcessTime;

    /**
     * 处理时间（毫秒）
     */
    private long clsProcessTime;

    /**
     * 旋转归正的检测框个数
     */
    private int clsRotBox;

    /**
     * 处理时间（毫秒）
     */
    private long recProcessTime;

    /**
     * 是否成功
     */
    private boolean success;

    /**
     * 错误信息
     */
    private String error;

    /**
     * 检测框处理结果
     */
    private List<TextBox> boxes;
}
