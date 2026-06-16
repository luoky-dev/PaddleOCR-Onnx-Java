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
     * 原图像
     */
    private Mat rawMat;

    /**
     * 原图像名
     */
    private String imageName;

    /**
     * 图像检测运行状态
     */
    private DetState detState;

    /**
     * 图像检测结果检测框
     */
    private List<TextBox> detResultBoxes;

    /**
     * 分类检测运行状态
     */
    private List<ClsBatch> clsBatches;

    /**
     * 分类检测结果检测框
     */
    private List<TextBox> clsResultBoxes;

    /**
     * 图像识别运行状态
     */
    private List<RecBatch> recBatches;

    /**
     * 图像识别结果检测框
     */
    private List<TextBox> recResultBoxes;
}
