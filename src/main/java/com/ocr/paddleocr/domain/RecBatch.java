package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Rec识别阶段批次处理信息
 */
@Data
@Builder
@AllArgsConstructor
public final class RecBatch {

    /**
     * 当前处理批次内的文本框
     */
    private final List<TextBox> boxes;

    /**
     * 预处理后的 CHW 数组
     */
    private final List<float[]> chwList;

    /**
     * 当前批次内的统一宽度
     */
    private final int batchWidth;

    /**
     * 当前批次的推理结果
     */
    private float[][][] probs;
}
