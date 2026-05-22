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
     * 检测框顺序
     */
    private int index;

    /**
     * 宽高比, 用于cls和rec模型批量输入
     */
    private double aspectRatio;

    /**
     * 顶点, 检测到的四边形检测框顶点
     */
    private List<Point> points;

    /**
     * 透视变换裁剪后的矩形检测框图像
     */
    private Mat cropMat;

    /**
     * 分类模型处理旋转后图像信息
     */
    private Mat rotMat;

    /**
     * 原检测框方向角度
     */
    private int angle;

    /**
     * 方向分类置信度
     */
    private float clsConfidence;

    /**
     * 是否旋转
     */
    private boolean isRotate;

    /**
     * 旋转的角度
     */
    private int rotAngle;

    /**
     * 识别的文本内容
     */
    private String recText;

    /**
     * 文本识别置信度
     */
    private float recConfidence;
}
