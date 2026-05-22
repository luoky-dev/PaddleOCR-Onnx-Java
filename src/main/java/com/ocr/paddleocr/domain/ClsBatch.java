package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.opencv.core.Size;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
public class ClsBatch {

    /**
     * 当前批次预处理后的 CHW 数组
     */
    private List<float[]> chwList;

    /**
     * 当前处理批次内的检测框
     */
    private List<TextBox> boxes;

    /**
     * 当前批次内输入模型尺寸
     */
    private Size modelInputSize;

    /**
     * 当前批次的cls分类模型推理结果
     */
    private float[][] probVector;
}
