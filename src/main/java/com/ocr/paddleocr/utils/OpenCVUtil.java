package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * OpenCV 工具类
 * 提供图像处理常用操作的封装，支持自动资源管理
 */
public class OpenCVUtil {

    /**
     * 获取边界矩形
     *
     * @param points 四个顶点坐标
     * @return Rect 矩形
     */
    public static Rect getBoundingRect(List<Point> points) {
        if (points == null || points.isEmpty()) {
            return new Rect(0, 0, 0, 0);
        }
        double minX = points.stream().mapToDouble(p -> p.x).min().orElse(0);
        double minY = points.stream().mapToDouble(p -> p.y).min().orElse(0);
        double maxX = points.stream().mapToDouble(p -> p.x).max().orElse(0);
        double maxY = points.stream().mapToDouble(p -> p.y).max().orElse(0);
        return new Rect((int) minX, (int) minY,
                (int) (maxX - minX), (int) (maxY - minY));
    }

    /**
     * 计算两个矩形的IoU（交并比）
     *
     * @param a 矩形A
     * @param b 矩形B
     * @return IoU值 (0-1)
     */
    public static float computeIoU(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);
        if (x2 <= x1 || y2 <= y1) {
            return 0;
        }
        int interArea = (x2 - x1) * (y2 - y1);
        int areaA = a.width * a.height;
        int areaB = b.width * b.height;
        int unionArea = areaA + areaB - interArea;
        return (float) interArea / unionArea;
    }

    /**
     * 透视变换裁剪
     *
     * @param image 原始图像
     * @param box 文本框四点坐标
     * @return 校正后的矩形图像
     */
    public static Mat perspectiveTransformCrop(Mat image, List<Point> box) {
        // 获取四点坐标并排序
        Point[] srcPoints = box.toArray(new Point[4]);
        Point[] sortedSrc = orderPoints(srcPoints);

        // 计算目标矩形的宽度和高度
        double width = Math.max(
                distance(sortedSrc[0], sortedSrc[1]),
                distance(sortedSrc[2], sortedSrc[3])
        );
        double height = Math.max(
                distance(sortedSrc[0], sortedSrc[3]),
                distance(sortedSrc[1], sortedSrc[2])
        );
        width = Math.max(width, 1);
        height = Math.max(height, 1);

        // 目标矩形四点坐标
        Point[] dstPoints = {
                new Point(0, 0),
                new Point(width - 1, 0),
                new Point(width - 1, height - 1),
                new Point(0, height - 1)
        };

        // 计算透视变换矩阵
        MatOfPoint2f srcMat = new MatOfPoint2f(sortedSrc);
        MatOfPoint2f dstMat = new MatOfPoint2f(dstPoints);
        Mat transform = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        // 执行透视变换
        Mat result = MatPipeline.fromMat(image)
                .warpPerspective(transform, new Size(width, height))
                .get();
        // 释放资源
        srcMat.release();
        dstMat.release();
        transform.release();
        return result;
    }

    /**
     * 矩形裁剪
     *
     * @param image 原始图像
     * @param box 文本框四点坐标
     * @return 裁剪后的图像
     */
    public static Mat rectangleCrop(Mat image, List<Point> box) {
        // 计算最小外接矩形
        Rect boundingRect = Imgproc.boundingRect(new MatOfPoint(box.toArray(new Point[0])));
        // 边界检查
        boundingRect.x = Math.max(0, boundingRect.x);
        boundingRect.y = Math.max(0, boundingRect.y);
        boundingRect.width = Math.min(boundingRect.width, image.cols() - boundingRect.x);
        boundingRect.height = Math.min(boundingRect.height, image.rows() - boundingRect.y);
        if (boundingRect.width <= 0 || boundingRect.height <= 0) {
            return null;
        }
        return new Mat(image, boundingRect);
    }

    /**
     * 计算两点间距离
     */
    public static double distance(Point p1, Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * 将四点坐标按顺时针顺序排序（从左上角开始）
     *
     * @param pts 四点坐标数组
     * @return 排序后的四点坐标数组 [左上, 右上, 右下, 左下]
     */
    public static Point[] orderPoints(Point[] pts) {
        if (pts == null || pts.length != 4) {
            return pts;
        }
        // 计算中心点
        double cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4;
        double cy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4;
        // 按角度排序
        List<Point> points = new ArrayList<>(Arrays.asList(pts));
        points.sort(Comparator.comparingDouble(p -> Math.atan2(p.y - cy, p.x - cx)));
        return points.toArray(new Point[4]);
    }

    /**
     * 将四点坐标按顺时针顺序排序（从左上角开始）
     *
     * @param pts 四点坐标列表
     * @return 排序后的四点坐标列表 [左上, 右上, 右下, 左下]
     */
    public static List<Point> orderPoints(List<Point> pts) {
        if (pts == null || pts.size() != 4) {
            return pts;
        }
        // 计算中心点
        double cx = pts.stream().mapToDouble(p -> p.x).sum() / 4;
        double cy = pts.stream().mapToDouble(p -> p.y).sum() / 4;
        // 按角度排序
        List<Point> ordered = new ArrayList<>(pts);
        ordered.sort(Comparator.comparingDouble(p -> Math.atan2(p.y - cy, p.x - cx)));
        return ordered;
    }

}