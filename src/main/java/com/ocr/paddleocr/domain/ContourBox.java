package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.opencv.core.Point;
import org.opencv.core.Rect;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContourBox {

    /**
     * 轮廓框原始顶点
     */
    private Point[] points;

    /**
     * 轮廓框面积
     */
    private double area;

    /**
     * 边界框尺寸
     */
    private Rect boundingRect;

    /**
     * 噪声框过滤标志
     */
    private boolean noiseFilter;

    /**
     * 多边近似后的顶点
     */
    private Point[] approxPoints;

    /**
     * 多边近似失败标志(true: 多边近似后顶点数量异常)
     */
    private boolean approxFilter;

    /**
     * 四边拟合后的顶点
     */
    private Point[] quadPoints;

    /**
     * 排序后的顶点
     */
    private Point[] orderPoints;

    /**
     * 平均置信度
     */
    private double score;

    /**
     * 置信度过低过滤标志(true: 小于最低阈值触发过滤)
     */
    private boolean scoreFilter;

    /**
     * 扩张后轮廓顶点
     */
    private Point[] unclipPoints;

    /**
     * 宽高比
     */
    private double aspectRatio;

    /**
     * 还原后在原图的坐标
     */
    private Point[] restorePoints;
}
