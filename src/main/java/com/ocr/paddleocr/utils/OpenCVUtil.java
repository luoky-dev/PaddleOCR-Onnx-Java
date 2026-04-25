package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * OpenCV 工具类
 */
public class OpenCVUtil {


    // 预定义颜色（BGR格式）
    private static final Scalar COLOR_RED = new Scalar(0, 0, 255);
    private static final Scalar COLOR_GREEN = new Scalar(0, 255, 0);
    private static final Scalar COLOR_BLUE = new Scalar(255, 0, 0);
    private static final Scalar COLOR_YELLOW = new Scalar(0, 255, 255);
    private static final Scalar COLOR_CYAN = new Scalar(255, 255, 0);
    private static final Scalar COLOR_MAGENTA = new Scalar(255, 0, 255);
    private static final Scalar COLOR_WHITE = new Scalar(255, 255, 255);
    private static final Scalar COLOR_BLACK = new Scalar(0, 0, 0);

    /**
     * 在图像上绘制检测框
     *
     * @param image 原始图像
     * @param boxes 检测框坐标列表
     * @param color 框颜色
     * @param thickness 线条粗细
     * @return 绘制后的图像（克隆，不影响原图）
     */
    public static Mat drawBoxes(Mat image, List<List<Point>> boxes, Scalar color, int thickness) {
        if (image == null || image.empty()) {
            return null;
        }

        if (boxes == null || boxes.isEmpty()) {
            return image.clone();
        }

        // 克隆图像，避免修改原图
        Mat result = image.clone();

        for (int i = 0; i < boxes.size(); i++) {
            List<Point> box = boxes.get(i);
            if (box == null || box.size() < 4) {
                continue;
            }

            // 绘制多边形
            drawPolygon(result, box, color, thickness);

            // 可选：绘制序号
            Point center = getCenter(box);
            putText(result, String.valueOf(i + 1), center, color);
        }

        return result;
    }

    /**
     * 在图像上绘制检测框（使用默认颜色和粗细）
     */
    public static Mat drawBoxes(Mat image, List<List<Point>> boxes) {
        return drawBoxes(image, boxes, COLOR_GREEN, 2);
    }

    /**
     * 绘制多边形
     */
    public static void drawPolygon(Mat image, List<Point> points, Scalar color, int thickness) {
        if (points == null || points.size() < 3) {
            return;
        }

        // 将点转换为 MatOfPoint
        MatOfPoint matOfPoint = new MatOfPoint();
        matOfPoint.fromList(points);

        // 绘制多边形轮廓
        Imgproc.polylines(image, java.util.Collections.singletonList(matOfPoint),
                true, color, thickness);

        OpenCVUtil.releaseMat(matOfPoint);
    }

    public static void drawPolygon(Mat image, List<Point> points) {
        drawPolygon(image, points, COLOR_GREEN, 2);
    }

    public static void drawText(Mat image, String text, Point position, Scalar color, double fontScale, int thickness) {
        if (image == null || image.empty() || text == null || position == null) {
            return;
        }
        Imgproc.putText(image, text, position, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);
    }

    public static void drawText(Mat image, String text, Point position) {
        drawText(image, text, position, COLOR_GREEN,0.5, 1);
    }

    /**
     * 绘制矩形框（轴对齐）
     */
    public static Mat drawRects(Mat image, List<Rect> rects, Scalar color, int thickness) {
        if (image == null || image.empty()) {
            return null;
        }

        Mat result = image.clone();

        for (Rect rect : rects) {
            Imgproc.rectangle(result, rect.tl(), rect.br(), color, thickness);
        }

        return result;
    }

    /**
     * 绘制旋转矩形框
     */
    public static Mat drawRotatedRects(Mat image, List<RotatedRect> rects, Scalar color, int thickness) {
        if (image == null || image.empty()) {
            return null;
        }

        Mat result = image.clone();

        for (RotatedRect rect : rects) {
            Point[] vertices = new Point[4];
            rect.points(vertices);

            // 将顶点转换为列表
            List<Point> points = java.util.Arrays.asList(vertices);
            drawPolygon(result, points, color, thickness);
        }

        return result;
    }

    /**
     * 获取多边形中心点
     */
    private static Point getCenter(List<Point> points) {
        if (points == null || points.isEmpty()) {
            return new Point(0, 0);
        }

        double sumX = 0, sumY = 0;
        for (Point p : points) {
            sumX += p.x;
            sumY += p.y;
        }
        return new Point(sumX / points.size(), sumY / points.size());
    }

    /**
     * 在图像上添加文本
     */
    private static void putText(Mat image, String text, Point position, Scalar color) {
        // 添加背景矩形使文字更清晰
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thicknessText = 1;

        // 使用数组来模拟引用传递
        int[] baseline = new int[1];
        Size textSize = Imgproc.getTextSize(text, fontFace, fontScale, thicknessText, baseline);

        // 绘制背景矩形
        int bgX = (int) position.x - 2;
        int bgY = (int) position.y - (int) textSize.height - 2;
        int bgWidth = (int) textSize.width + 4;
        int bgHeight = (int) textSize.height + baseline[0] + 4;

        Rect bgRect = new Rect(bgX, bgY, bgWidth, bgHeight);
        Imgproc.rectangle(image, bgRect.tl(), bgRect.br(), COLOR_BLACK, -1);

        // 绘制文字
        Point textPos = new Point(position.x, position.y);
        Imgproc.putText(image, text, textPos, fontFace, fontScale, color, thicknessText);
    }

    /**
     * 根据置信度获取颜色
     */
    private static Scalar getColorByConfidence(float confidence) {
        if (confidence >= 0.8) {
            return COLOR_GREEN;      // 高置信度 - 绿色
        } else if (confidence >= 0.5) {
            return COLOR_YELLOW;     // 中置信度 - 黄色
        } else {
            return COLOR_RED;        // 低置信度 - 红色
        }
    }

    /**
     * 保存绘制结果到文件
     */
    public static void saveImage(Mat image, String outputPath) {
        if (image == null || image.empty()) {
            throw new IllegalArgumentException("Image cannot be empty");
        }

        File outputFile = new File(outputPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists() && !parentDir.mkdirs()) {
            throw new IllegalStateException("Failed to create parent directory: " + parentDir.getAbsolutePath());
        }

        Imgcodecs.imwrite(outputPath, image);
    }

    public static Mat getImage(String inputPath){
        if (inputPath == null || inputPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Path cannot be empty");
        }
        Mat image = Imgcodecs.imread(inputPath);
        if (image.empty()) {
            throw new IllegalArgumentException("Unable to read image");
        }
        return image;
    }

    public static List<String> readDictionary(String dictPath) {
        List<String> dictionary = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(dictPath), StandardCharsets.UTF_8))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine && line.startsWith("\uFEFF")) {
                    line = line.substring(1);
                }
                firstLine = false;
                if (line.isEmpty()) {
                    continue;
                }
                dictionary.add(line);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to read dictionary: " + dictPath, e);
        }
        return dictionary;
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
        Mat result = new Mat();
        Imgproc.warpPerspective(image, result, transform, new Size(width, height));
        // 释放资源
        releaseMat(srcMat);
        releaseMat(dstMat);
        releaseMat(transform);
        return result;
    }

    /**
     * 多边形裁剪（适用于 detUsePolygon=true 的不规则文本框）
     * 流程：多边形掩码 -> 位与保留区域 -> 外接矩形裁剪
     *
     * @param image 原始图像
     * @param polygon 多边形顶点（>=3）
     * @return 裁剪后的图像，失败时返回 empty Mat
     */
    public static Mat polygonCrop(Mat image, List<Point> polygon) {
        if (image == null || image.empty() || polygon == null || polygon.size() < 3) {
            return new Mat();
        }

        int maxX = image.cols() - 1;
        int maxY = image.rows() - 1;
        if (maxX < 0 || maxY < 0) {
            return new Mat();
        }

        // 顶点裁剪到图像范围内，避免 fillPoly / boundingRect 越界问题
        List<Point> clipped = new ArrayList<>(polygon.size());
        for (Point p : polygon) {
            double x = Math.max(0, Math.min(p.x, maxX));
            double y = Math.max(0, Math.min(p.y, maxY));
            clipped.add(new Point(x, y));
        }

        MatOfPoint poly = new MatOfPoint();
        poly.fromList(clipped);
        Rect rect = Imgproc.boundingRect(poly);
        if (rect.width <= 0 || rect.height <= 0) {
            releaseMat(poly);
            return new Mat();
        }

        Mat mask = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);
        Imgproc.fillPoly(mask, Collections.singletonList(poly), new Scalar(255));

        Mat masked = new Mat();
        Core.bitwise_and(image, image, masked, mask);
        Mat cropped = new Mat(masked, rect).clone();

        releaseMat(poly);
        releaseMat(mask);
        releaseMat(masked);
        return cropped;
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
            releaseMat(ch);
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
        releaseMat(resized);

        // 转换为 HWC 格式数组
        // HWC: Height x Width x Channel (高度 x 宽度 x 通道)
        float[] hwc = new float[imgH * resizedW * 3];
        floatMat.get(0, 0, hwc);
        releaseMat(floatMat);

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
     * 按宽高两个缩放比例分别还原坐标到原图坐标
     *
     * @param points 还原前坐标
     * @param scaleX 宽度缩放比例
     * @param scaleY 高度缩放比例
     * @param srcW 原图宽度
     * @param srcH 原图高度
     * @return List<Point>
     */
    public static List<Point> restorePoints(List<Point> points, float scaleX, float scaleY, int srcW, int srcH) {
        float safeScaleX = scaleX <= 0 ? 1.0f : scaleX;
        float safeScaleY = scaleY <= 0 ? 1.0f : scaleY;
        List<Point> restored = new ArrayList<>();
        for (Point p : points) {
            double x = p.x / safeScaleX;
            double y = p.y / safeScaleY;
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
        releaseMat(matOfPoint);

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

        releaseMat(mask);
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

            // 扩张向量
            moveVecs.add(new double[]{vx * distance, ux * distance});
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

        releaseMat(mat);
        releaseMat(approx);

        return result;
    }

    /**
     * Mat 资源判空释放
     */
    public static Mat buildProbMat(float[][] probMap) {
        if (probMap == null || probMap.length == 0 || probMap[0].length == 0) {
            return new Mat();
        }
        int h = probMap.length;
        int w = probMap[0].length;
        Mat mat = new Mat(h, w, CvType.CV_32FC1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                mat.put(y, x, probMap[y][x]);
            }
        }
        return mat;
    }

    public static Mat createProbHeatmap(float[][] probMap) {
        Mat prob = buildProbMat(probMap);
        if (prob.empty()) {
            releaseMat(prob);
            return new Mat();
        }
        Mat prob8 = new Mat();
        Core.normalize(prob, prob8, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
        Mat heatmap = new Mat();
        Imgproc.applyColorMap(prob8, heatmap, Imgproc.COLORMAP_JET);
        releaseMat(prob8);
        releaseMat(prob);
        return heatmap;
    }

    public static Mat createBinaryMap(float[][] probMap, float threshold) {
        Mat prob = buildProbMat(probMap);
        if (prob.empty()) {
            releaseMat(prob);
            return new Mat();
        }
        Mat binary = new Mat();
        Imgproc.threshold(prob, binary, threshold, 255, Imgproc.THRESH_BINARY);
        binary.convertTo(binary, CvType.CV_8UC1);
        releaseMat(prob);
        return binary;
    }

    public static Mat toVisualizableImage(Mat src, boolean srcIsRgb) {
        if (src == null || src.empty()) {
            return new Mat();
        }
        Mat vis;
        if (src.depth() == CvType.CV_8U) {
            vis = src.clone();
        } else {
            vis = new Mat();
            Core.normalize(src, vis, 0, 255, Core.NORM_MINMAX);
            vis.convertTo(vis, CvType.CV_8UC(src.channels()));
        }

        if (vis.channels() == 1) {
            Mat bgr = new Mat();
            Imgproc.cvtColor(vis, bgr, Imgproc.COLOR_GRAY2BGR);
            releaseMat(vis);
            vis = bgr;
        } else if (vis.channels() == 4) {
            Mat bgr = new Mat();
            Imgproc.cvtColor(vis, bgr, Imgproc.COLOR_BGRA2BGR);
            releaseMat(vis);
            vis = bgr;
        }

        if (srcIsRgb && vis.channels() == 3) {
            Mat bgr = new Mat();
            Imgproc.cvtColor(vis, bgr, Imgproc.COLOR_RGB2BGR);
            releaseMat(vis);
            vis = bgr;
        }
        return vis;
    }

    public static Mat resizeToHeight(Mat src, int targetHeight) {
        if (src == null || src.empty() || targetHeight <= 0) {
            return new Mat();
        }
        if (src.rows() == targetHeight) {
            return src.clone();
        }
        int targetWidth = Math.max(1, (int) Math.round((double) src.cols() * targetHeight / src.rows()));
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(targetWidth, targetHeight));
        return resized;
    }

    public static Mat concatHorizontal(List<Mat> mats) {
        if (mats == null || mats.isEmpty()) {
            return new Mat();
        }
        Mat merged = new Mat();
        Core.hconcat(mats, merged);
        return merged;
    }

    public static void saveImageAndRelease(Mat image, String outputPath) {
        if (image == null || image.empty()) {
            releaseMat(image);
            return;
        }
        saveImage(image, outputPath);
        releaseMat(image);
    }

    public static void ensureDir(String dir) {
        File file = new File(dir);
        if (!file.exists() && !file.mkdirs()) {
            throw new IllegalStateException("failed to create debug directory: " + dir);
        }
    }

    public static void releaseMat(Mat mat) {
        if (mat != null) {
            mat.release();
        }
    }
}
