package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OCRContext {

    /**
     * 处理前的图像数据
     */
    private Mat rawMat;

    /**
     * 检测预处理后的图像数据
     */
    private Mat detPrepMat;

    /**
     * 检测预处理缩放比例
     */
    private float detPrepScale;

    /**
     * 检测模型输出的概率图
     */
    private float[][] detProbMap;

    /**
     * 检测模型后处理过滤后轮廓检测框
     */
    private List<List<Point>> detContourBoxes;

    /**
     * 检测模型后处理还原并裁剪后的检测框
     */
    private List<List<Point>> detPostBoxes;

    /**
     * 检测模型处理时间（毫秒）
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
