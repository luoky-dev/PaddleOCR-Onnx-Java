package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.opencv.core.Size;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
public class DetState {

    /**
     * 原图像尺寸
     */
    private Size rawMatSize;

    /**
     * 缩放后的图像尺寸
     */
    private Size resizeMatSize;

    /**
     * 检测模型输入尺寸
     */
    private Size modelInputSize;

    /**
     * 检测模型输入的CHW格式数据
     */
    private float[] chwData;

    /**
     * 检测模型输出的概率图
     */
    private float[][] probMap;

    /**
     * 检测出的轮廓框
     */
    private List<ContourBox> contourBoxes;
}
