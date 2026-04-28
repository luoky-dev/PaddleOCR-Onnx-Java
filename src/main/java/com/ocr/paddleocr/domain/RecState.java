package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Rec识别状态
 */
@Data
@Builder
@AllArgsConstructor
public final class RecState {

    /**
     * 原始顺序, 保证识别结果顺序
     */
    private final List<TextBox> originalOrder;

    /**
     * 所有检测框识别批次
     */
    private final List<RecBatch> batches;

    /**
     * Rec模型输入固定高度
     */
    private final int recHeight;
}
