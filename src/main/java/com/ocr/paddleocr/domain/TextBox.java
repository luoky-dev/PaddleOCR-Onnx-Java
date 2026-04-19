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
public class TextBox {

    /**
     * 文本框图像信息
     */
    private Mat rawMat;

    /**
     * 分类模型处理旋转后图像信息
     */
    private Mat rotMat;

    /**
     * 模型输出
     */
    private float[][] modelOutput;

    /**
     * 检测框四个顶点
     */
    private List<Point> box;

    /**
     * 原检测框方向角度
     */
    private int angle;

    /**
     * 方向分类置信度
     */
    private float clsConfidence;

    /**
     * 文本识别置信度
     */
    private float recConfidence;

    /**
     * 是否旋转
     */
    private boolean isRotate;

    /**
     * 旋转后的角度
     */
    private int rotAngle;

    /**
     * 方向分类旋转后的角度
     */
    private int clsAngle;

    /**
     * 识别的文本内容
     */
    private String text;
}
