package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * OpenCV 资源管理工具类
 *
 * @author OCR Team
 * @version 1.1
 */
public class OpenCVResource {

    // ==================== 基础资源管理 ====================

    /**
     * 安全执行操作，自动释放资源
     */
    public static <T extends Mat> void use(T resource, Consumer<T> action) {
        try {
            action.accept(resource);
        } finally {
            release(resource);
        }
    }

    /**
     * 安全执行操作并返回结果，自动释放资源
     */
    public static <T extends Mat, R> R map(T resource, Function<T, R> mapper) {
        try {
            return mapper.apply(resource);
        } finally {
            release(resource);
        }
    }

    /**
     * 释放单个 Mat 资源
     */
    public static void release(Mat resource) {
        if (resource != null && !resource.empty()) {
            resource.release();
        }
    }

    /**
     * 释放多个 Mat 资源
     */
    public static void release(Mat... resources) {
        for (Mat resource : resources) {
            release(resource);
        }
    }

    /**
     * 释放轮廓列表
     */
    public static void release(List<? extends Mat> resources) {
        if (resources == null) return;
        for (Mat resource : resources) {
            release(resource);
        }
    }

    // ==================== 轮廓相关 ====================

    /**
     * 查找轮廓（自动释放 hierarchy）
     */
    public static List<MatOfPoint> findContours(Mat binaryMat) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        try {
            Imgproc.findContours(binaryMat, contours, hierarchy,
                    Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            return contours;
        } finally {
            release(hierarchy);
        }
    }

    /**
     * 查找轮廓并处理（自动释放所有资源）
     */
    public static void withContours(Mat binaryMat, Consumer<List<MatOfPoint>> processor) {
        List<MatOfPoint> contours = findContours(binaryMat);
        try {
            processor.accept(contours);
        } finally {
            release(contours);
        }
    }

    // ==================== 多边形近似相关 ====================

    /**
     * 多边形近似，返回 Point 列表（自动释放所有临时资源）
     */
    public static List<Point> approxPolyDP(Point[] points, double epsilon, boolean closed) {
        MatOfPoint2f verticesMat = null;
        MatOfPoint2f approxCurve = null;
        try {
            verticesMat = new MatOfPoint2f(points);
            approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(verticesMat, approxCurve, epsilon, closed);

            List<Point> result = new ArrayList<>();
            for (int i = 0; i < approxCurve.total(); i++) {
                double[] point = approxCurve.get(i, 0);
                result.add(new Point(point[0], point[1]));
            }
            return result;
        } finally {
            release(verticesMat, approxCurve);
        }
    }

    // ==================== 最小外接矩形相关 ====================

    /**
     * 计算点集的最小外接矩形（自动释放临时资源）
     */
    public static RotatedRect minAreaRect(Point[] points) {
        MatOfPoint2f mat = null;
        try {
            mat = new MatOfPoint2f(points);
            return Imgproc.minAreaRect(mat);
        } finally {
            release(mat);
        }
    }

    /**
     * 从轮廓计算最小外接矩形（自动释放临时资源）
     */
    public static RotatedRect minAreaRectFromContour(MatOfPoint contour) {
        if (contour == null || contour.empty()) {
            return null;
        }

        Point[] points = contour.toArray();
        if (points.length < 5) {
            // 点数不足，无法计算有效的最小外接矩形
            return null;
        }

        return map(contour, c -> {
            MatOfPoint2f contour2f = new MatOfPoint2f(c.toArray());
            try {
                // 检查轮廓2f是否有效
                if (contour2f.empty() || contour2f.total() < 5) {
                    return null;
                }
                return Imgproc.minAreaRect(contour2f);
            } catch (Exception e) {
                System.err.println("minAreaRect 计算失败: " + e.getMessage());
                return null;
            } finally {
                release(contour2f);
            }
        });
    }

    // ==================== 弧长/周长计算 ====================

    /**
     * 计算点集的弧长/周长（自动释放临时资源）
     */
    public static double arcLength(Point[] points, boolean closed) {
        MatOfPoint2f mat = null;
        try {
            mat = new MatOfPoint2f(points);
            return Imgproc.arcLength(mat, closed);
        } finally {
            release(mat);
        }
    }

    /**
     * 计算轮廓周长
     */
    public static double arcLength(MatOfPoint contour, boolean closed) {
        return map(contour, c -> Imgproc.arcLength(new MatOfPoint2f(c.toArray()), closed));
    }

    // ==================== 轮廓面积计算 ====================

    /**
     * 计算轮廓面积（安全版本，不释放传入的轮廓）
     */
    public static double contourArea(MatOfPoint contour) {
        if (contour == null || contour.empty()) return 0;
        return Imgproc.contourArea(contour);
    }

    // ==================== 旋转矩形顶点获取 ====================

    /**
     * 获取旋转矩形的四个顶点
     */
    public static Point[] getRotatedRectPoints(RotatedRect rect) {
        Point[] points = new Point[4];
        rect.points(points);
        return points;
    }

    // ==================== 边界框计算 ====================

    /**
     * 计算轮廓的边界框
     */
    public static Rect boundingRect(MatOfPoint contour) {
        return map(contour, c -> Imgproc.boundingRect(c));
    }

    /**
     * 计算点集的边界框
     */
    public static Rect boundingRect(Point[] points) {
        if (points == null || points.length == 0) {
            return new Rect(0, 0, 0, 0);
        }
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;
        for (Point p : points) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }
        return new Rect((int) minX, (int) minY,
                (int) (maxX - minX), (int) (maxY - minY));
    }


    /**
     * 多边形扩张（Unclip）- 完全对齐PaddleOCR官方实现
     *
     * @param vertices  原始多边形顶点（按顺序排列）
     * @param distance  扩张距离（正数向外扩张，负数向内收缩）
     * @return 扩张后的多边形顶点列表
     */
    public static List<Point> unclipPolygon(Point[] vertices, double distance) {
        if (vertices == null || vertices.length < 3) {
            return vertices == null ? new ArrayList<>() : Arrays.asList(vertices);
        }

        if (Math.abs(distance) < 1e-6) {
            return Arrays.asList(vertices);
        }

        int n = vertices.length;

        // 1. 计算每条边的外扩向量
        List<double[]> moveVecs = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            Point p1 = vertices[i];
            Point p2 = vertices[(i + 1) % n];

            // 计算边的方向向量
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double length = Math.hypot(dx, dy);

            if (length < 1e-6) {
                moveVecs.add(new double[]{0, 0});
                continue;
            }

            // 单位方向向量
            double ux = dx / length;
            double uy = dy / length;

            // 垂直向量（向外）
            double vx = -uy;
            double vy = ux;

            // 扩张向量
            moveVecs.add(new double[]{vx * distance, vy * distance});
        }

        // 2. 计算新顶点位置
        List<Point> expanded = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            double[] move1 = moveVecs.get(i);
            double[] move2 = moveVecs.get((i + 1) % n);

            Point p = vertices[(i + 1) % n];

            // 两条边的扩张向量之和
            double newX = p.x + move1[0] + move2[0];
            double newY = p.y + move1[1] + move2[1];

            expanded.add(new Point(newX, newY));
        }

        return expanded;
    }

    /**
     * 多边形扩张（使用轮廓面积和周长计算距离）- PaddleOCR标准方法
     *
     * 官方算法：
     * distance = area * unclip_ratio / perimeter
     *
     * @param contour      轮廓点集
     * @param unclipRatio  扩张比例（官方默认2.5）
     * @param minArea      最小面积阈值
     * @return 扩张后的多边形顶点列表，如果不满足条件返回null
     */
    public static List<Point> unclipContour(MatOfPoint contour, float unclipRatio, double minArea) {
        if (contour == null || contour.empty()) {
            return null;
        }

        double area = contourArea(contour);
        if (area < minArea) {
            return null;
        }

        // 计算周长
        double perimeter = arcLength(contour, true);
        if (perimeter < 1e-6) {
            return null;
        }

        // 计算扩张距离：area * unclip_ratio / perimeter
        double distance = area * unclipRatio / perimeter;

        // 获取轮廓顶点
        Point[] points = contour.toArray();

        // 执行扩张
        return unclipPolygon(points, distance);
    }

    /**
     * 多边形扩张的另一种实现（基于旋转矩形）
     * 适用于矩形文本框的快速扩张
     *
     * @param rect         旋转矩形
     * @param unclipRatio  扩张比例
     * @return 扩张后的四点坐标
     */
    public static List<Point> unclipRotatedRect(RotatedRect rect, float unclipRatio) {
        Point[] vertices = getRotatedRectPoints(rect);

        // 计算矩形面积和周长
        double area = rect.size.area();
        double perimeter = 2 * (rect.size.width + rect.size.height);
        double distance = area * unclipRatio / perimeter;

        // 扩张矩形（向四个方向扩展）
        List<Point> expanded = new ArrayList<>();
        for (Point p : vertices) {
            // 计算从中心到顶点的方向
            double dx = p.x - rect.center.x;
            double dy = p.y - rect.center.y;
            double len = Math.hypot(dx, dy);
            if (len > 1e-6) {
                double scale = 1 + distance / len;
                expanded.add(new Point(
                        rect.center.x + dx * scale,
                        rect.center.y + dy * scale
                ));
            } else {
                expanded.add(p);
            }
        }

        return expanded;
    }

    // ==================== 完整的文本框提取流程 ====================

    /**
     * 从轮廓提取文本框（完整流程）
     * 包含：面积过滤 → 最小外接矩形 → unclip扩张 → 多边形近似
     *
     * @param contour      轮廓
     * @param unclipRatio  扩张比例（PaddleOCR默认2.5）
     * @param minArea      最小面积阈值（官方默认3）
     * @return 文本框点集（4个点），如果不满足条件返回 null
     */
    public static List<Point> extractTextbox(MatOfPoint contour, float unclipRatio, double minArea) {
        if (contour == null || contour.empty()) {
            return null;
        }

        double area = contourArea(contour);
        if (area < minArea) {
            return null;
        }

        Point[] contourPoints = contour.toArray();
        if (contourPoints.length < 4) {
            return null;
        }

        // 方法1: 直接使用最小外接矩形（不扩张，先测试）
        try {
            MatOfPoint2f contour2f = new MatOfPoint2f(contourPoints);
            RotatedRect rect = Imgproc.minAreaRect(contour2f);
            contour2f.release();

            // 获取矩形顶点
            Point[] vertices = getRotatedRectPoints(rect);

            // 如果矩形太小，跳过
            if (rect.size.width < 5 || rect.size.height < 5) {
                return null;
            }

            // 尝试扩张
            double perimeter = arcLength(contour, true);
            if (perimeter > 1e-6) {
                double distance = area * unclipRatio / perimeter;
                if (distance > 0 && distance < 100) {  // 限制最大扩张距离
                    List<Point> expanded = unclipPolygon(vertices, distance);
                    if (expanded != null && expanded.size() >= 4) {
                        vertices = expanded.toArray(new Point[0]);
                    }
                }
            }

            // 多边形近似
            double epsilon = 0.01 * arcLength(vertices, true);
            List<Point> box = approxPolyDP(vertices, epsilon, true);

            if (box.size() == 4) {
                return box;
            }

            // 如果近似后不是4点，直接使用矩形顶点
            if (vertices.length == 4) {
                return Arrays.asList(vertices);
            }

        } catch (Exception e) {
            System.err.println("extractTextbox error: " + e.getMessage());
        }

        return null;
    }

    /**
     * 计算两点间距离
     */
    private static double distance(Point p1, Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}