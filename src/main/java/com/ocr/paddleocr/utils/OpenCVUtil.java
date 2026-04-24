package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * OpenCV 工具类
 */
public class OpenCVUtil {

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
        Mat result = new Mat();
        Imgproc.warpPerspective(image, result, transform, new Size(width, height));
        // 释放资源
        srcMat.release();
        dstMat.release();
        transform.release();
        return result;
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

    /**
     * RGB图像的归一化 + 标准化
     * @param rgb RGB通道
     * @param mean RGB通道的均值
     * @param std RGB通道的标准差
     * @return Mat
     */
    public static Mat normalize(Mat rgb, float[] mean, float[] std) {
        // 转换为浮点并归一化x
        Mat floatMat = new Mat();
        // 将 0-255 的整数值转换为 0.0-1.0 的浮点数
        rgb.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);

        // 分离RGB三个通道
        List<Mat> channels = new ArrayList<>();
        Core.split(floatMat, channels);

        // 标准化 (Standardization), 公式: (x - mean) / std
        // 使数据分布接近标准正态分布, 有助于模型收敛
        for (int i = 0; i < 3; i++) {
            // 减去均值
            Core.subtract(channels.get(i), new Scalar(mean[i]), channels.get(i));
            // 除以标准差
            Core.divide(channels.get(i), new Scalar(std[i]), channels.get(i));
        }
        // 合并通道
        Core.merge(channels, floatMat);
        // 释放临时资源
        for (Mat ch : channels) {
            ch.release();
        }
        return floatMat;
    }

    /**
     * 将图像缩放归一化到 [-1, 1], 并转换为 CHW 格式
     *
     * @param mat 原始文本图像
     * @param imgH 模型要求的高度
     * @param imgW 模型要求的宽度
     * @return CHW格式的浮点数组 [3, imgH, imgW], 值范围 [-1, 1]
     */
    public static float[] resizeNormalize(Mat mat, int imgH, int imgW){

        // 计算缩放后的宽度（保持高宽比）
        int srcH = mat.rows();
        int srcW = mat.cols();
        // 计算宽高比
        float ratio = srcH > 0 ? (float) srcW / (float) srcH : 1.0f;

        // 计算缩放后的宽度, 高度固定为 imgH, 宽度按比例缩放
        int resizedW = Math.min(imgW, Math.max(1, Math.round(imgH * ratio)));

        // 缩放图像到, 保持文本不扭曲
        Mat resized = new Mat();
        Imgproc.resize(mat, resized, new Size(resizedW, imgH));

        // 转换为 float32 类型, 并归一化到 [0, 1]
        Mat floatMat = new Mat();
        // 1.0/255.0 将像素值从 [0, 255] 映射到 [0, 1]
        resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);
        resized.release();

        // 转换为 HWC 格式数组
        // HWC: Height x Width x Channel (高度 x 宽度 x 通道)
        float[] hwc = new float[imgH * resizedW * 3];
        floatMat.get(0, 0, hwc);
        floatMat.release();

        // 转换为 ONNX 模型要求的输入 CHW 格式, 并归一化到 [-1, 1]
        // CHW: Channel x Height x Width (通道 x 高度 x 宽度)
        float[] chw = new float[3 * imgH * imgW];

        // 遍历：通道 → 高度 → 宽度
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < imgH; h++) {
                for (int w = 0; w < imgW; w++) {
                    // 计算CHW数组的索引
                    int chwIdx = (c * imgH + h) * imgW + w;
                    if (w < resizedW) {
                        // 图像区域：转换 HWC → CHW
                        // HWC索引: (行 × 宽度 + 列) × 3 + 通道
                        int hwcIdx = (h * resizedW + w) * 3 + c;
                        // 从 [0, 1] 归一化到 [-1, 1] 公式：output = (input - 0.5) / 0.5
                        chw[chwIdx] = (hwc[hwcIdx] - 0.5f) / 0.5f;
                    } else {
                        // Padding区域填充 -1 对应像素值 0（全黑）
                        chw[chwIdx] = -1.0f;
                    }
                }
            }
        }
        return chw;
    }

    /**
     * 还原坐标到原图坐标
     * @param points 还原前坐标
     * @param scale 缩放比例
     * @param srcW 原图像宽度
     * @param srcH 原图像高度
     * @return List<Point>
     */
    public static List<Point> restorePoints(List<Point> points, float scale, int srcW, int srcH) {
        // 防止 scale 无效（如 0 或负数）
        float safeScale = scale <= 0 ? 1.0f : scale;
        List<Point> restored = new ArrayList<>();
        for (Point p : points) {
            // 坐标还原：除以缩放比例
            double x = p.x / safeScale;
            double y = p.y / safeScale;
            // 边界裁剪，防止超出图像范围
            restored.add(new Point(
                    Math.max(0, Math.min(x, srcW - 1)),
                    Math.max(0, Math.min(y, srcH - 1))));
        }
        return restored;
    }

    /**
     * 计算轮廓内平均置信度
     *
     * @param contour 轮廓
     * @param probMap 概率图
     * @return 平均置信度
     */
    public static double getScore(MatOfPoint contour, float[][] probMap) {
        Point[] points = contour.toArray();
        if (points.length < 3) {
            return 0.0;
        }

        int height = probMap.length;
        int width = probMap[0].length;

        // 创建掩码
        Mat mask = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
        MatOfPoint matOfPoint = new MatOfPoint(points);
        Imgproc.fillPoly(mask, java.util.Collections.singletonList(matOfPoint), new Scalar(255));
        matOfPoint.release();

        // 计算轮廓内像素的平均概率
        double sum = 0;
        int count = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (mask.get(i, j)[0] > 0) {
                    sum += probMap[i][j];
                    count++;
                }
            }
        }

        mask.release();
        return count > 0 ? sum / count : 0.0;
    }

    /**
     * 多边形扩张算法
     *
     * @param polygon 原始多边形顶点
     * @param distance 扩张距离
     * @return 扩张后的多边形顶点
     */
    public static List<Point> unclipPolygon(Point[] polygon, double distance) {
        if (polygon == null || polygon.length < 3) {
            return new ArrayList<>();
        }

        if (Math.abs(distance) < 1e-6) {
            List<Point> result = new ArrayList<>();
            for (Point p : polygon) {
                result.add(p.clone());
            }
            return result;
        }

        int n = polygon.length;

        // 1. 计算每条边的外扩向量
        List<double[]> moveVecs = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            Point p1 = polygon[i];
            Point p2 = polygon[(i + 1) % n];

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

            Point p = polygon[(i + 1) % n];

            // 两条边的扩张向量之和
            double newX = p.x + move1[0] + move2[0];
            double newY = p.y + move1[1] + move2[1];

            expanded.add(new Point(newX, newY));
        }

        return expanded;
    }

    /**
     * 多边形近似算法
     *
     * @param points 原始点集
     * @param epsilon 近似精度
     * @param closed 是否闭合
     * @return 近似后的点集
     */
    public static List<Point> approxPolyDP(List<Point> points, double epsilon, boolean closed) {
        if (points == null || points.isEmpty()) {
            return new ArrayList<>();
        }

        MatOfPoint2f mat = new MatOfPoint2f();
        mat.fromList(points);

        MatOfPoint2f approx = new MatOfPoint2f();
        Imgproc.approxPolyDP(mat, approx, epsilon, closed);

        List<Point> result = new ArrayList<>();
        for (int i = 0; i < approx.total(); i++) {
            double[] point = approx.get(i, 0);
            result.add(new Point(point[0], point[1]));
        }

        mat.release();
        approx.release();

        return result;
    }

}