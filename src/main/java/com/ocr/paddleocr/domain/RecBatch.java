package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.opencv.core.Size;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
public class RecBatch {

    /**
     * 当前批次预处理后的 CHW 数组
     */
    private List<float[]> chwList;

    /**
     * 当前处理批次内的文本框
     */
    private List<TextBox> boxes;

    /**
     * 当前批次内输入模型尺寸
     */
    private Size modelInputSize;

    /**
     * 当前批次的推理结果
     */
    private float[][][] prob;
}
