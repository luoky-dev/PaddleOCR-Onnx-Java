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
     * 处理前的原图像
     */
    private Mat rawMat;

    /**
     * 整图旋转90度后的图像(适用于[0,180]的二分类cls模型)
     * 当整图检测框宽高比 < 1 的框占比过高会触发整图旋转重新检测
     */
    private Mat rotMat;

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
     * cls或rec预处理分组
     */
    private List<ClsBatch> clsBatches;

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
     * 分类检测处理时间（毫秒）
     */
    private long clsProcessTime;

    /**
     * 识别模型处理结果检测框
     */
    private List<TextBox> recResultBoxes;

    /**
     * 识别模型处理时间（毫秒）
     */
    private long recProcessTime;

}
