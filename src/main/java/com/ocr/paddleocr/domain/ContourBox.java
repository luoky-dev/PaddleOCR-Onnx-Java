package com.ocr.paddleocr.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.opencv.core.Point;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContourBox {

    /**
     * 轮廓框顺序
     */
    private int index;

    /**
     * 轮廓框面积
     */
    private double area;

    /**
     * 面积过滤标志(true: 小于最低阈值触发过滤)
     */
    private boolean areaFilter;

    /**
     * 平均置信度
     */
    private double score;

    /**
     * 平均置信度过滤标志(true: 小于最低阈值触发过滤)
     */
    private boolean scoreFilter;

    /**
     * 扩张后轮廓顶点
     */
    private Point[] unclipPoints;

    /**
     * 扩张失败标志(true: 扩张后顶点数量异常, 扩张失败)
     */
    private boolean unclipFail;

    /**
     * 多边近似后的顶点
     */
    private Point[] approxPoints;

    /**
     * 多边近似失败标志(true: 多边近似后顶点数量异常, 启用算法失败)
     */
    private boolean approxFail;

    /**
     * 最小尺寸
     */
    private double minSize;

    /**
     * 最小尺寸过滤(true: 小于最低阈值触发过滤)
     */
    private boolean minSizeFilter;

    /**
     * 宽高比
     */
    private double aspectRatio;

    /**
     * 宽高比过滤(true: 大于最大阈值触发过滤)
     */
    private boolean aspectRatioFilter;

    /**
     * 还原后在原图的坐标
     */
    private Point[] restorePoints;
}
