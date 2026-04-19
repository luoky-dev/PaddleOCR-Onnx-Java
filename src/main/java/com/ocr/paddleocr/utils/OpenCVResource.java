package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * OpenCV 资源管理工具类
 * <p>
 * 统一管理 OpenCV 各种类型（Mat、MatOfPoint、MatOfPoint2f、RotatedRect 等）的资源，
 * 提供自动释放机制，避免内存泄漏。
 * </p>
 *
 * <h3>使用示例：</h3>
 * <pre>{@code
 * // 1. 处理轮廓
 * OpenCVResource.withContours(binaryMat, contours -> {
 *     for (MatOfPoint contour : contours) {
 *         double area = Imgproc.contourArea(contour);
 *         // 处理轮廓...
 *     }
 * });
 *
 * // 2. 多边形近似
 * List<Point> box = OpenCVResource.approxPolyDP(points, epsilon, true);
 *
 * // 3. 计算最小外接矩形
 * RotatedRect rect = OpenCVResource.minAreaRect(points);
 *
 * // 4. 查找轮廓并返回结果
 * List<MatOfPoint> contours = OpenCVResource.findContours(binaryMat);
 * try {
 *     // 使用 contours...
 * } finally {
 *     OpenCVResource.release(contours);
 * }
 * }</pre>
 *
 * @author OCR Team
 * @version 1.0
 */

public class OpenCVResource {

    // ==================== 基础资源管理 ====================

    /**
     * 安全执行操作，自动释放资源
     *
     * @param resource 需要管理的资源
     * @param action   操作
     * @param <T>      资源类型（必须是 Mat 的子类）
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
     *
     * @param resource 需要管理的资源
     * @param mapper   映射函数
     * @param <T>      资源类型（必须是 Mat 的子类）
     * @param <R>      返回类型
     * @return 映射结果
     */
    public static <T extends Mat, R> R map(T resource, Function<T, R> mapper) {
        try {
            return mapper.apply(resource);
        } finally {
            release(resource);
        }
    }

    /**
     * 安全执行操作，自动释放多个资源
     *
     * @param action   操作
     * @param resources 需要管理的资源
     */
    public static void useAll(Consumer<Mat[]> action, Mat... resources) {
        try {
            action.accept(resources);
        } finally {
            release(resources);
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
     *
     * @param binaryMat 二值图像
     * @return 轮廓列表（调用者负责释放）
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
     *
     * @param binaryMat 二值图像
     * @param processor 轮廓处理器
     */
    public static void withContours(Mat binaryMat, Consumer<List<MatOfPoint>> processor) {
        List<MatOfPoint> contours = findContours(binaryMat);
        try {
            processor.accept(contours);
        } finally {
            release(contours);
        }
    }

    /**
     * 查找轮廓并返回处理结果（自动释放所有资源）
     *
     * @param binaryMat 二值图像
     * @param processor 轮廓处理器
     * @param <R>       返回类型
     * @return 处理结果
     */
    public static <R> R mapContours(Mat binaryMat, Function<List<MatOfPoint>, R> processor) {
        List<MatOfPoint> contours = findContours(binaryMat);
        try {
            return processor.apply(contours);
        } finally {
            release(contours);
        }
    }

    /**
     * 处理单个轮廓（自动释放）
     *
     * @param contour   轮廓
     * @param processor 处理器
     */
    public static void withContour(MatOfPoint contour, Consumer<MatOfPoint> processor) {
        try {
            processor.accept(contour);
        } finally {
            release(contour);
        }
    }

    // ==================== 多边形近似相关 ====================

    /**
     * 多边形近似，返回 Point 列表（自动释放所有临时资源）
     *
     * @param points  原始点集
     * @param epsilon 近似精度
     * @param closed  是否闭合
     * @return 近似后的点列表
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

    /**
     * 多边形近似并处理结果（自动释放）
     *
     * @param points  原始点集
     * @param epsilon 近似精度
     * @param closed  是否闭合
     * @param action  处理近似结果的操作
     */
    public static void useApproxPolyDP(Point[] points, double epsilon, boolean closed,
                                       Consumer<List<Point>> action) {
        List<Point> result = approxPolyDP(points, epsilon, closed);
        try {
            action.accept(result);
        } finally {
            // result 是 Point 列表，不需要释放
        }
    }

    // ==================== 最小外接矩形相关 ====================

    /**
     * 计算点集的最小外接矩形（自动释放临时资源）
     *
     * @param points 点集
     * @return 最小外接矩形
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
     *
     * @param contour 轮廓
     * @return 最小外接矩形
     */
    public static RotatedRect minAreaRectFromContour(MatOfPoint contour) {
        return map(contour, c -> {
            MatOfPoint2f contour2f = new MatOfPoint2f(c.toArray());
            try {
                return Imgproc.minAreaRect(contour2f);
            } finally {
                release(contour2f);
            }
        });
    }

    // ==================== 弧长/周长计算 ====================

    /**
     * 计算点集的弧长/周长（自动释放临时资源）
     *
     * @param points 点集
     * @param closed 是否闭合
     * @return 弧长
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

    // ==================== 轮廓面积计算 ====================

    /**
     * 计算轮廓面积（安全版本，不释放传入的轮廓）
     *
     * @param contour 轮廓
     * @return 面积
     */
    public static double contourArea(MatOfPoint contour) {
        if (contour == null || contour.empty()) return 0;
        return Imgproc.contourArea(contour);
    }

    // ==================== 旋转矩形顶点获取 ====================

    /**
     * 获取旋转矩形的四个顶点
     *
     * @param rect 旋转矩形
     * @return 四个顶点
     */
    public static Point[] getRotatedRectPoints(RotatedRect rect) {
        Point[] points = new Point[4];
        rect.points(points);
        return points;
    }

    // ==================== 图像矩计算 ====================

    /**
     * 计算轮廓的矩（自动释放临时资源）
     *
     * @param contour 轮廓
     * @return 矩
     */
    public static Moments moments(MatOfPoint contour) {
        return map(contour, c -> Imgproc.moments(c));
    }

    // ==================== 凸包计算 ====================

    /**
     * 计算轮廓的凸包（自动释放临时资源）
     *
     * @param contour 轮廓
     * @return 凸包点集（调用者负责释放）
     */
    public static MatOfPoint convexHull(MatOfPoint contour) {
        MatOfInt hull = new MatOfInt();
        try {
            Imgproc.convexHull(contour, hull);
            // 将索引转换为实际点
            Point[] points = contour.toArray();
            int[] hullIndices = hull.toArray();
            Point[] hullPoints = new Point[hullIndices.length];
            for (int i = 0; i < hullIndices.length; i++) {
                hullPoints[i] = points[hullIndices[i]];
            }
            return new MatOfPoint(hullPoints);
        } finally {
            release(hull);
        }
    }

    /**
     * 计算轮廓的凸包并处理（自动释放所有资源）
     *
     * @param contour 轮廓
     * @param action  处理凸包的操作
     */
    public static void withConvexHull(MatOfPoint contour, Consumer<MatOfPoint> action) {
        MatOfPoint hull = convexHull(contour);
        try {
            action.accept(hull);
        } finally {
            release(hull);
        }
    }

    // ==================== 轮廓周长计算 ====================

    /**
     * 计算轮廓周长
     *
     * @param contour 轮廓
     * @param closed  是否闭合
     * @return 周长
     */
    public static double arcLength(MatOfPoint contour, boolean closed) {
        return map(contour, c -> Imgproc.arcLength(new MatOfPoint2f(c.toArray()), closed));
    }

    // ==================== 边界框计算 ====================

    /**
     * 计算轮廓的边界框
     *
     * @param contour 轮廓
     * @return 边界框
     */
    public static Rect boundingRect(MatOfPoint contour) {
        return map(contour, c -> Imgproc.boundingRect(c));
    }

    /**
     * 计算点集的边界框
     *
     * @param points 点集
     * @return 边界框
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

    // ==================== 点是否在轮廓内 ====================

    /**
     * 判断点是否在轮廓内
     *
     * @param contour 轮廓
     * @param point   点
     * @return 是否在内部
     */
    public static boolean pointInContour(MatOfPoint2f contour, Point point) {
        return map(contour, c -> Imgproc.pointPolygonTest(c, point, false) >= 0);
    }

    // ==================== 轮廓匹配 ====================

    /**
     * 计算两个轮廓的匹配度
     *
     * @param contour1 轮廓1
     * @param contour2 轮廓2
     * @param method   匹配方法
     * @return 匹配度
     */
    public static double matchShapes(MatOfPoint contour1, MatOfPoint contour2, int method) {
        MatOfPoint2f c1 = null, c2 = null;
        try {
            c1 = new MatOfPoint2f(contour1.toArray());
            c2 = new MatOfPoint2f(contour2.toArray());
            return Imgproc.matchShapes(c1, c2, method, 0.0);
        } finally {
            release(c1, c2);
        }
    }

    // ==================== 便捷的组合方法 ====================

    /**
     * 从轮廓提取文本框（完整流程）
     *
     * @param contour     轮廓
     * @param unclipRatio 扩张比例
     * @param minArea     最小面积阈值
     * @return 文本框点集，如果不满足条件返回 null
     */
    public static List<Point> extractTextbox(MatOfPoint contour, float unclipRatio, double minArea) {
        double area = contourArea(contour);
        if (area < minArea) {
            return null;
        }

        // 计算最小外接矩形
        RotatedRect rect = minAreaRectFromContour(contour);
        Point[] vertices = getRotatedRectPoints(rect);

        // 计算扩张距离
        double unclipDist = calculateUnclipDistance(rect.size, unclipRatio);
        if (unclipDist > 0) {
            List<Point> expanded = unclipPolygon(vertices, unclipDist);
            if (expanded.size() >= 4) {
                vertices = expanded.toArray(new Point[0]);
            }
        }

        // 多边形近似
        double epsilon = 0.01 * arcLength(vertices, true);
        List<Point> box = approxPolyDP(vertices, epsilon, true);

        // 确保是4个点
        if (box.size() == 4) {
            return box;
        }
        return null;
    }

    // 辅助方法（需要根据实际实现）
    private static double calculateUnclipDistance(Size size, float ratio) {
        double area = size.width * size.height;
        return Math.sqrt(area) * ratio;
    }

    private static List<Point> unclipPolygon(Point[] vertices, double distance) {
        // 实现多边形扩张逻辑
        // 这里只是一个示例，需要根据实际算法实现
        List<Point> result = new ArrayList<>();
        for (Point p : vertices) {
            result.add(p);
        }
        return result;
    }
}