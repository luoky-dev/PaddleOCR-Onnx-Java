package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.opencv.core.Mat;

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
     * 检测模型处理结果检测框
     */
    private List<TextBox> detResultBoxes;

    /**
     * 检测模型处理时间（毫秒）
     */
    private long detProcessTime;

    /**
     * 分类检测分批处理检测框
     */
    private List<List<TextBox>> clsBatchBoxes;

    /**
     * 分类检测模型分批预处理后的模型输入数据
     */
    private List<List<float[]>> clsBatchChw;

    /**
     * 分类检测模型输出 logits 数组
     */
    private List<float[][]> clsLogitsList;

    /**
     * 分类检测模型处理结果检测框
     */
    private List<TextBox> clsResultBoxes;

    /**
     * 旋转归正的检测框个数
     */
    private int clsRotBox;

    /**
     * 分类检测处理时间（毫秒）
     */
    private long clsProcessTime;

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


}
