package com.ocr.paddleocr.process;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * PaddleOCR 官方 DB 后处理实现
 */
@Slf4j
public class DBPostProcess {

    // 官方默认参数
    public static final float THRESH = 0.2f;           // 二值化阈值
    public static final float BOX_THRESH = 0.6f;       // 框置信度阈值
    public static final float UNCLIP_RATIO = 2.0f;     // 扩张比例
    public static final int MIN_AREA = 5;             // 最小面积
    public static final float NMS_IOU_THRESH = 0.3f;   // NMS IoU阈值
    public static final int EXPAND_KERNEL_SIZE = 3;    // 膨胀核大小

    /**
     * 官方 expand_bitmap 方法
     * 膨胀二值图并使用概率图过滤
     *
     * @param bitmap 二值图 (boolean数组)
     * @param probMap 概率图
     * @param boxThresh 框阈值
     * @return 膨胀后的二值图
     */
    public static boolean[][] expandBitmap(boolean[][] bitmap, float[][] probMap, float boxThresh) {
        int height = bitmap.length;
        int width = bitmap[0].length;

        // 1. 将 boolean 数组转换为 Mat
        Mat binaryMat = new Mat(height, width, CvType.CV_8UC1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                binaryMat.put(i, j, bitmap[i][j] ? 255 : 0);
            }
        }

        // 2. 膨胀操作 (3x3 核)
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(EXPAND_KERNEL_SIZE, EXPAND_KERNEL_SIZE));
        Mat dilated = new Mat();
        Imgproc.dilate(binaryMat, dilated, kernel);
        kernel.release();

        // 3. 创建概率图阈值掩码 (pred > box_thresh)
        Mat probMask = new Mat(height, width, CvType.CV_8UC1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                probMask.put(i, j, probMap[i][j] > boxThresh ? 255 : 0);
            }
        }

        // 4. 膨胀结果与掩码进行与操作
        Mat expanded = new Mat();
        Core.bitwise_and(dilated, probMask, expanded);

        // 5. 转换回 boolean 数组
        boolean[][] result = new boolean[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double[] val = expanded.get(i, j);
                result[i][j] = val[0] > 0;
            }
        }

        // 释放资源
        binaryMat.release();
        dilated.release();
        probMask.release();
        expanded.release();

        return result;
    }

    /**
     * 从二值图查找轮廓
     *
     * @param bitmap 二值图 (boolean数组)
     * @return 轮廓列表
     */
    public static List<MatOfPoint> findContours(boolean[][] bitmap) {
        int height = bitmap.length;
        int width = bitmap[0].length;

        // 转换为 Mat
        Mat binaryMat = new Mat(height, width, CvType.CV_8UC1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                binaryMat.put(i, j, bitmap[i][j] ? 255 : 0);
            }
        }
        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binaryMat, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // 释放资源
        binaryMat.release();
        hierarchy.release();

        return contours;
    }

    /**
     * 官方 unclip 多边形扩张
     *
     * @param rect 最小外接矩形
     * @param distance 扩张距离
     * @return 扩张后的多边形点集
     */
    public static List<Point> unclip(RotatedRect rect, double distance) {
        // 获取矩形四个顶点
        Point[] vertices = new Point[4];
        rect.points(vertices);
        return unclipPolygon(vertices, distance);
    }

    /**
     * 官方 _unclip_polygon 多边形扩张核心算法
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
     * 官方多边形近似 (approxPolyDP)
     *
     * @param points 原始点集
     * @param epsilon 近似精度 (官方: 0.002 * perimeter)
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

    /**
     * 官方 get_score 计算轮廓内平均置信度
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
     * 官方 boxes_to_xyxy 转换
     * 将多边形框转换为 (x1, y1, x2, y2, score) 格式
     *
     * @param boxes 多边形框列表
     * @param scores 置信度列表
     * @return xyxy格式的框
     */
    public static List<float[]> boxesToXyxy(List<List<Point>> boxes, List<Float> scores) {
        List<float[]> result = new ArrayList<>();

        for (int i = 0; i < boxes.size(); i++) {
            List<Point> box = boxes.get(i);
            float[] xyxy = new float[5];

            float minX = Float.MAX_VALUE;
            float minY = Float.MAX_VALUE;
            float maxX = Float.MIN_VALUE;
            float maxY = Float.MIN_VALUE;

            for (Point p : box) {
                minX = Math.min(minX, (float) p.x);
                minY = Math.min(minY, (float) p.y);
                maxX = Math.max(maxX, (float) p.x);
                maxY = Math.max(maxY, (float) p.y);
            }

            xyxy[0] = minX;
            xyxy[1] = minY;
            xyxy[2] = maxX;
            xyxy[3] = maxY;
            xyxy[4] = scores.get(i);

            result.add(xyxy);
        }

        return result;
    }

    /**
     * 官方 NMS (使用 OpenCV DNN)
     *
     * @param boxes 多边形框列表
     * @param scores 置信度列表
     * @param scoreThreshold 分数阈值
     * @param nmsThreshold NMS IoU阈值
     * @return NMS后的索引
     */
    public static List<Integer> nmsBoxes(List<List<Point>> boxes, List<Float> scores,
                                         float scoreThreshold, float nmsThreshold) {
        if (boxes.isEmpty()) {
            return new ArrayList<>();
        }

        // 转换为 xyxy 格式
        List<Rect> rects = new ArrayList<>();
        List<Float> scoreList = new ArrayList<>();

        for (int i = 0; i < boxes.size(); i++) {
            if (scores.get(i) > scoreThreshold) {
                Rect rect = getBoundingRect(boxes.get(i));
                rects.add(rect);
                scoreList.add(scores.get(i));
            }
        }

        if (rects.isEmpty()) {
            return new ArrayList<>();
        }

        // 方法1：使用 MatOfRect2d（某些版本需要）
        MatOfRect2d rects2d = new MatOfRect2d();
        List<Rect2d> rect2dList = new ArrayList<>();
        for (Rect rect : rects) {
            rect2dList.add(new Rect2d(rect.x, rect.y, rect.width, rect.height));
        }
        rects2d.fromList(rect2dList);

        MatOfFloat scoresMat = new MatOfFloat();
        scoresMat.fromList(scoreList);

        MatOfInt indicesMat = new MatOfInt();

        // 调用 OpenCV DNN NMS
        Dnn.NMSBoxes(rects2d, scoresMat, scoreThreshold, nmsThreshold, indicesMat);

        int[] indices = indicesMat.toArray();
        List<Integer> result = new ArrayList<>();
        for (int idx : indices) {
            result.add(idx);
        }

        rects2d.release();
        scoresMat.release();
        indicesMat.release();

        return result;
    }

    private static Rect getBoundingRect(List<Point> points) {
        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE;
        double maxY = Double.MIN_VALUE;

        for (Point p : points) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }

        return new Rect((int) minX, (int) minY, (int) (maxX - minX), (int) (maxY - minY));
    }

    /**
     * 官方 restore_boxes 坐标还原
     *
     * @param boxes 检测框
     * @param scale 缩放比例
     * @return 还原后的检测框
     */
    public static List<List<Point>> restoreBoxes(List<List<Point>> boxes, float scale) {
        List<List<Point>> restored = new ArrayList<>();

        for (List<Point> box : boxes) {
            List<Point> newBox = new ArrayList<>();
            for (Point p : box) {
                newBox.add(new Point(p.x / scale, p.y / scale));
            }
            restored.add(newBox);
        }

        return restored;
    }
}