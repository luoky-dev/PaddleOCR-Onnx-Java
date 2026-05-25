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
     * 图像检测阶段运行状态
     */
    private DetState detState;

    /**
     * 图像检测阶段结果检测框
     */
    private List<TextBox> detResultBoxes;

    /**
     * cls或rec预处理分组
     */
    private List<ClsBatch> clsBatches;

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
