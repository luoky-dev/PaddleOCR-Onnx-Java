package com.ocr.paddleocr.utils;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * OpenCV 工具类
 */
public class OpenCVUtil {


    // 预定义颜色（BGR格式）
    public static final Scalar COLOR_RED = new Scalar(0, 0, 255);
    public static final Scalar COLOR_GREEN = new Scalar(0, 255, 0);
    public static final Scalar COLOR_BLUE = new Scalar(255, 0, 0);
    public static final Scalar COLOR_YELLOW = new Scalar(0, 255, 255);
    public static final Scalar COLOR_CYAN = new Scalar(255, 255, 0);
    public static final Scalar COLOR_MAGENTA = new Scalar(255, 0, 255);
    public static final Scalar COLOR_WHITE = new Scalar(255, 255, 255);
    public static final Scalar COLOR_BLACK = new Scalar(0, 0, 0);

    /**
     * 在图像上绘制检测框
     *
     * @param image 原始图像
     * @param boxes 检测框坐标列表
     * @param color 框颜色
     * @param thickness 线条粗细
     * @return 绘制后的图像（克隆，不影响原图）
     */
    public static Mat drawBoxes(Mat image, List<Point[]> boxes, Scalar color, int thickness) {
        if (image == null || image.empty()) {
            return null;
        }

        if (boxes == null || boxes.isEmpty()) {
            return image.clone();
        }

        // 克隆图像，避免修改原图
        Mat result = image.clone();

        for (int i = 0; i < boxes.size(); i++) {
            Point[] box = boxes.get(i);
            if (box == null || box.length < 4) {
                continue;
            }

            // 绘制多边形
            drawPolygon(result, box, color, thickness);

            // 可选：绘制序号
//            Point center = getCenter(box);
//            putText(result, String.valueOf(i + 1), center, color);
        }

        return result;
    }

    /**
     * 在图像上绘制检测框（使用默认颜色和粗细）
     */
    public static Mat drawBoxes(Mat image, List<Point[]> boxes) {
        return drawBoxes(image, boxes, COLOR_GREEN, 2);
    }

    /**
     * 绘制多边形
     */
    public static void drawPolygon(Mat image, Point[] points, Scalar color, int thickness) {

        // 将点转换为 MatOfPoint
        MatOfPoint matOfPoint = new MatOfPoint();
        matOfPoint.fromArray(points);

        // 绘制多边形轮廓
        Imgproc.polylines(image, Collections.singletonList(matOfPoint),
                true, color, thickness);

        OpenCVUtil.releaseMat(matOfPoint);
    }

    public static void drawPolygon(Mat image, Point[] points) {
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
            drawPolygon(result, vertices, color, thickness);
        }

        return result;
    }

    /**
     * 获取多边形中心点
     */
    private static Point getCenter(Point[] points) {
        if (points == null) {
            return new Point(0, 0);
        }

        double sumX = 0, sumY = 0;
        for (Point p : points) {
            sumX += p.x;
            sumY += p.y;
        }
        return new Point(sumX / points.length, sumY / points.length);
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
     * 保存图像
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

    /**
     * 读取图像
     */
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

    /**
     * 获取图像名
     */
    public static String getImageName(String inputPath){
        if (inputPath == null || inputPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Path cannot be empty");
        }
        File file = new File(inputPath);
        if (!file.exists()) {
            throw new IllegalArgumentException("Unable to read image");
        }
        return file.getName();
    }

    /**
     * 读取字典
     */
    public static String[] readDictionary(String dictPath) throws IOException {
        List<String> dictList = new ArrayList<>();
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(dictPath), StandardCharsets.UTF_8));
        String line;
        // 开始添加background token
        dictList.add("");
        while ((line = br.readLine()) != null) {
            dictList.add(line);
        }
        // 末尾添加unknown token
        dictList.add(" ");
        br.close();
        return dictList.toArray(new String[0]);
    }

    /**
     * 透视变换裁剪
     *
     * @param image 原始图像
     * @param points 文本框四点坐标
     * @return 裁剪的矩形图像
     */
    public static Mat perspectiveTransformCrop(Mat image, Point[] points) {

        // 计算目标矩形的宽度和高度
        double width = Math.max(
                distance(points[0], points[1]),
                distance(points[2], points[3])
        );
        double height = Math.max(
                distance(points[0], points[3]),
                distance(points[1], points[2])
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
        MatOfPoint2f srcMat = new MatOfPoint2f(points);
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
     * 将指定区域置空
     * @param srcMat 原始图像
     * @param points 区域顶点坐标
     */
    public static void fillPolyWhite(Mat srcMat, Point[] points) {
        if (srcMat == null || points == null || points.length < 3) {
            return;
        }

        // 1. 将顶点转换为MatOfPoint
        MatOfPoint matOfPoint = new MatOfPoint(points);
        List<MatOfPoint> polygons = List.of(matOfPoint);

        // 2. 填充白色
        Imgproc.fillPoly(srcMat, polygons, COLOR_WHITE);

        // 3. 释放资源
        matOfPoint.release();
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
     * 对四点坐标按顺时针/逆时针排序（基于中心点角度）
     * @param points 四个点的数组
     * @return 排序后的四个点数组（按角度从 -π 到 π 排序）
     */
    public static Point[] orderPoints(Point[] points) {
        if (points == null || points.length != 4) {
            return points;
        }

        // 1. 计算中心点
        double cx = (points[0].x + points[1].x + points[2].x + points[3].x) / 4;
        double cy = (points[0].y + points[1].y + points[2].y + points[3].y) / 4;

        // 2. 按角度排序
        List<Point> orderPoints = new ArrayList<>(Arrays.asList(points));
        orderPoints.sort(Comparator.comparingDouble(p -> Math.atan2(p.y - cy, p.x - cx)));

        // 3. 将起始点旋转到左上角
        // 计算每个点的 x+y 值，最小的通常是左上角
        int leftTopIndex = 0;
        double minSum = orderPoints.get(0).x + orderPoints.get(0).y;
        for (int i = 1; i < 4; i++) {
            Point p = orderPoints.get(i);
            double sum = p.x + p.y;
            if (sum < minSum) {
                minSum = sum;
                leftTopIndex = i;
            }
        }

        // 4. 重新排列，从左上角开始
        Point[] result = new Point[4];
        for (int i = 0; i < 4; i++) {
            result[i] = orderPoints.get((leftTopIndex + i) % 4);
        }

        return result;
    }

    /**
     * 交换四点顺序，实现宽高交换（竖排转横排）
     * @param points 原始四点坐标（已排序：左上、右上、右下、左下）
     * @return 交换后的四点坐标
     */
    public static Point[] rotateOrderPoints(Point[] points) {
        if (points == null || points.length != 4) {
            return points;
        }

        // 原始顺序: [0]左上, [1]右上, [2]右下, [3]左下
        // 交换后: [0]左上, [1]左下, [2]右下, [3]右上
        return new Point[]{
                points[0],  // 左上保持不变
                points[3],  // 左下 -> 右上
                points[2],  // 右下保持不变
                points[1]   // 右上 -> 左下
        };
    }

    /**
     * 长边限制和对齐到步长倍数方法
     * @param srcSize 原图尺寸
     * @param limitSize 长边限制尺寸
     * @param strideSize 步长尺寸
     * @return 缩放尺寸
     */
    public static Size longSideLimitToStride(Size srcSize, int limitSize, int strideSize) {
        int srcW = (int) srcSize.width;
        int srcH = (int) srcSize.height;
        // 1. 基于长边计算缩放比例
        int maxSide = Math.max(srcW, srcH);
        float scale = maxSide > limitSize ? (float) limitSize / maxSide : 1.0f;

        // 2. 等比例缩放
        int scaledW = Math.max(Math.round(srcW * scale), 1);
        int scaledH = Math.max(Math.round(srcH * scale), 1);

        // 3. 向上对齐到指定倍数
        int dstW = (int) Math.ceil((double) scaledW / strideSize) * strideSize;
        int dstH = (int) Math.ceil((double) scaledH / strideSize) * strideSize;

        return new Size(Math.max(dstW, 1), Math.max(dstH, 1));
    }

    /**
     * 将尺寸中的宽度对齐到strideSize倍数
     * @param srcSize 原始尺寸
     * @param strideSize 对齐步长
     * @return 对齐后的尺寸（高度不变，宽度对齐）
     */
    public static Size widthToStride(Size srcSize, int strideSize) {
        int alignedWidth = ((int) srcSize.width + strideSize - 1) / strideSize * strideSize;
        return new Size(Math.max(alignedWidth, 1), Math.max(srcSize.height, 1));
    }

    /**
     * 等比例缩放到固定高度
     * @param srcSize 原始尺寸 (width, height)
     * @param targetHeight 目标高度
     * @return 缩放后的尺寸
     */
    public static Size getFixHeightSize(Size srcSize, int targetHeight) {
        int srcW = (int) srcSize.width;
        int srcH = (int) srcSize.height;
        // 计算缩放比例
        float scale = (float) targetHeight / srcH;
        // 计算缩放后的宽度
        int targetWidth = Math.round(srcW * scale);
        return new Size(Math.max(targetWidth, 1), Math.max(targetHeight, 1));
    }

    /**
     * 将图像缩放到目标宽高并转换RGB通道
     * @param mat 原图
     * @param dstSize 目标尺寸
     * @return mat 缩放转换RGB通道后的图像
     */
    public static Mat resizeToRGB(Mat mat, Size dstSize) {
        // 缩放图像，使用双线性插值，保持图像内容不变形
        Mat resized = new Mat();
        Imgproc.resize(mat, resized, dstSize);
        // 转换RGB通道
        Mat rgb = new Mat();
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        // 释放资源
        releaseMat(resized);
        return rgb;
    }

    /**
     * 图像填充, 左上对齐
     * @param mat 原图像
     * @param dstSize 目标尺寸
     * @return 填充后图像
     */
    public static Mat padding(Mat mat, Size dstSize){
        if (mat.width() > dstSize.width || mat.height() > dstSize.height) {
            return mat;
        }
        // 创建目标尺寸的黑色背景
        Mat result = new Mat(dstSize, mat.type());
        result.setTo(new Scalar(0, 0, 0));
        // 左上对齐放置
        Rect roi = new Rect(0, 0, mat.width(), mat.height());
        Mat roiMat = result.submat(roi);
        mat.copyTo(roiMat);
        releaseMat(roiMat);
        return result;
    }

    /**
     * 通用图像归一化转换CHW格式方法
     * @param mat 缩放转换RGB通道后的图像
     * @param mean 均值
     * @param std 标准差
     * @return CHW格式的float数组
     */
    public static float[] normalizeToCHW(Mat mat, float[] mean, float[] std) {
        int height = mat.rows();
        int width = mat.cols();
        int channels = mat.channels();
        // 1. 归一化到 [0,1]
        Mat floatMat = new Mat();
        mat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);
        // 2. 获取 HWC 格式数据
        float[] hwc = new float[height * width * channels];
        floatMat.get(0, 0, hwc);
        releaseMat(floatMat);
        // 3. 转换为 CHW 格式并应用归一化
        float[] chw = new float[channels * height * width];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int chwIdx = (c * height + h) * width + w;
                    int hwcIdx = (h * width + w) * channels + c;
                    // 应用归一化公式
                    chw[chwIdx] = (hwc[hwcIdx] - mean[c]) / std[c];
                }
            }
        }
        return chw;
    }

    /**
     * 通用解码方法
     * @param probs 概率数组
     * @return int[]{最大概率索引, 最大概率}
     */
    public static int[] decode(float[] probs){
        // 找出最大概率的索引
        int bestIndex = 0;
        float bestProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > bestProb) {
                bestProb = probs[i];
                bestIndex = i;
            }
        }
        // 将概率值通过 floatToIntBits 编码为 int 便于存储
        return new int[]{bestIndex, Float.floatToIntBits(bestProb)};
    }

    public static Mat rotate(Mat srcMat, int angle) {
        Mat dstMat = new Mat();
        if (angle == 180) {
            // 180度旋转
            Core.rotate(srcMat, dstMat, Core.ROTATE_180);
        } else if (angle == 90) {
            // 90度顺时针旋转
            Core.rotate(srcMat, dstMat, Core.ROTATE_90_CLOCKWISE);
        } else if (angle == 270) {
            // 90度逆时针旋转（等价于270度顺时针）
            Core.rotate(srcMat, dstMat, Core.ROTATE_90_COUNTERCLOCKWISE);
        } else {
            releaseMat(dstMat);
            return srcMat;
        }
        return dstMat;
    }

    /**
     * 还原检测框坐标到原图尺寸
     * @param points 当前图像上的顶点坐标数组
     * @param resizeSize 当前图像尺寸（缩放后的尺寸）
     * @param originalSize 原始图像尺寸
     * @return 还原后的坐标数组
     */
    public static Point[] restorePoints(Point[] points, Size resizeSize, Size originalSize) {
        // 计算缩放比例
        double scaleX =  originalSize.width / resizeSize.width;
        double scaleY =  originalSize.height / resizeSize.height;

        Point[] restored = new Point[points.length];
        for (int i = 0; i < points.length; i++) {
            Point p = points[i];
            double x = p.x * scaleX;
            double y = p.y * scaleY;

            // 裁剪到原图范围内
            x = Math.max(0, Math.min(x, originalSize.width - 1));
            y = Math.max(0, Math.min(y, originalSize.height - 1));

            restored[i] = new Point(x, y);
        }
        return restored;
    }

    /**
     * 计算轮廓内平均置信度
     *
     * @param contour 轮廓
     * @param probMat 概率图
     * @return 平均置信度
     */
    public static double getScore(MatOfPoint contour, Mat probMat) {
        if (contour == null || probMat == null || probMat.empty()) {
            return 0.0;
        }

        Point[] points = contour.toArray();
        if (points.length < 3) {
            return 0.0;
        }

        Rect rect = Imgproc.boundingRect(contour);
        if (rect.width <= 0 || rect.height <= 0) {
            return 0.0;
        }

        int safeX = Math.max(0, rect.x);
        int safeY = Math.max(0, rect.y);
        int safeWidth = Math.min(rect.width, probMat.cols() - safeX);
        int safeHeight = Math.min(rect.height, probMat.rows() - safeY);
        if (safeWidth <= 0 || safeHeight <= 0) {
            return 0.0;
        }

        Rect safeRect = new Rect(safeX, safeY, safeWidth, safeHeight);
        List<Point> roiPoints = new ArrayList<>(points.length);
        for (Point point : points) {
            roiPoints.add(new Point(point.x - safeRect.x, point.y - safeRect.y));
        }

        MatOfPoint roiContour = new MatOfPoint();
        roiContour.fromList(roiPoints);
        Mat mask = Mat.zeros(safeRect.height, safeRect.width, CvType.CV_8UC1);
        Imgproc.fillPoly(mask, Collections.singletonList(roiContour), new Scalar(255));

        Mat probRoi = new Mat(probMat, safeRect);
        Scalar mean = Core.mean(probRoi, mask);

        releaseMat(roiContour);
        releaseMat(mask);
        releaseMat(probRoi);
        return mean.val[0];
    }

    /**
     * 基于边平移的扩张
     * @param points 轮廓框顶点
     * @param unclipRatio 扩张比例
     * @return 扩张后顶点
     */
    public static Point[] unclip(Point[] points, double unclipRatio) {
        if (points == null || points.length != 4) {
            return points;
        }

        // 1. 计算尺寸、面积、周长
        double width = Math.max(
                distance(points[0], points[1]),
                distance(points[2], points[3])
        );
        double height = Math.max(
                distance(points[0], points[3]),
                distance(points[1], points[2])
        );
        double area = width * height;
        double perimeter = 2 * (width + height);

        // 2. 计算扩张距离（敏感于 unclipRatio）
        double distance = area * unclipRatio / perimeter;

        // 存储平移后的直线系数 [a, b, c]
        double[][] lines = new double[4][3];

        for (int i = 0; i < 4; i++) {
            Point p1 = points[i];
            Point p2 = points[(i + 1) % 4];

            // 计算法向量
            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double len = Math.hypot(dx, dy);

            double nx = -dy / len;
            double ny = dx / len;

            // 验证向外方向
            double cx = (p1.x + p2.x) / 2;
            double cy = (p1.y + p2.y) / 2;
            double centerX = (points[0].x + points[1].x + points[2].x + points[3].x) / 4;
            double centerY = (points[0].y + points[1].y + points[2].y + points[3].y) / 4;

            if (nx * (cx - centerX) + ny * (cy - centerY) < 0) {
                nx = -nx;
                ny = -ny;
            }

            // 平移后的直线: a*x + b*y = c
            lines[i][0] = nx;
            lines[i][1] = ny;
            lines[i][2] = nx * p1.x + ny * p1.y + distance;
        }

        // 计算交点
        Point[] expanded = new Point[4];
        for (int i = 0; i < 4; i++) {

            double[] l1 = lines[i];
            double[] l2 = lines[(i + 1) % 4];

            double det = l1[0] * l2[1] - l2[0] * l1[1];
            if (Math.abs(det) < 1e-6) {
                expanded[(i + 1) % 4] = points[(i + 1) % 4];
                continue;
            }

            double x = (l1[2] * l2[1] - l2[2] * l1[1]) / det;
            double y = (l1[0] * l2[2] - l2[0] * l1[2]) / det;

            expanded[(i + 1) % 4] = new Point(x, y);
        }

        return expanded;
    }

    /**
     * 通过矩形顶点获取矩形框尺寸/四边形最大尺寸
     * @param points 矩形四个顶点（已排序）
     * @return Size对象（最大宽度、最大高度）
     */
    public static Size getRectSize(Point[] points) {
        if (points == null || points.length != 4) {
            return new Size(0, 0);
        }

        // 计算宽度（取上边和下边的最大值）
        double width = Math.max(
                distance(points[0], points[1]),
                distance(points[2], points[3])
        );

        // 计算高度（取左边和右边的最大值）
        double height = Math.max(
                distance(points[0], points[3]),
                distance(points[1], points[2])
        );

        return new Size(Math.max(width, 1), Math.max(height, 1));
    }

    /**
     * 多边形近似算法
     * @param points 原始多边形顶点数组
     * @param epsilon 近似精度（越小越接近原形状，越大简化越多）
     * @param closed 是否为闭合多边形
     * @return 简化后的多边形顶点数组
     */
    public static Point[] approxPolyDP(MatOfPoint points, double epsilon, boolean closed) {
        if (points == null || points.toArray().length == 0) {
            return new Point[0];
        }

        // 转换为 MatOfPoint2f
        MatOfPoint2f mat = new MatOfPoint2f(points.toArray());
        // 计算周长
        double perimeter = epsilon * Imgproc.arcLength(mat, true);
        // 执行多边形近似
        MatOfPoint2f approx = new MatOfPoint2f();
        Imgproc.approxPolyDP(mat, approx, perimeter, closed);

        // 提取结果
        int total = (int) approx.total();
        Point[] result = new Point[total];
        for (int i = 0; i < total; i++) {
            double[] point = approx.get(i, 0);
            result[i] = new Point(point[0], point[1]);
        }
        // 释放资源
        releaseMat(approx);
        releaseMat(mat);

        return result;
    }

    public static Point[] minAreaRect(Point[] points) {
        // 最终过滤后返回的四边形顶点
        Point[] quadPoints = new Point[4];
        // 获取最小外接矩形顶点
        MatOfPoint2f approx2f = new MatOfPoint2f(points);
        RotatedRect rr = Imgproc.minAreaRect(approx2f);
        releaseMat(approx2f);
        // 设置顶点
        rr.points(quadPoints);
        return quadPoints;
    }

    /**
     * 检测框按阅读顺序排序（从左到右，从上到下）
     * @param points 检测框列表，每个框包含4个顶点坐标
     * @return 排序后的映射（位置索引 -> 四点坐标）
     */
    public static Map<Integer, Point[]> orderByRead(List<Point[]> points) {
        Map<Integer, Point[]> result = new LinkedHashMap<>();

        if (points == null || points.isEmpty()) {
            return result;
        }

        int n = points.size();

        // 存储每个框的边界信息
        double[] minX = new double[n];
        double[] maxX = new double[n];
        double[] minY = new double[n];
        double[] maxY = new double[n];

        for (int i = 0; i < n; i++) {
            Point[] box = points.get(i);
            minX[i] = Double.MAX_VALUE;
            maxX[i] = Double.MIN_VALUE;
            minY[i] = Double.MAX_VALUE;
            maxY[i] = Double.MIN_VALUE;

            for (Point p : box) {
                minX[i] = Math.min(minX[i], p.x);
                maxX[i] = Math.max(maxX[i], p.x);
                minY[i] = Math.min(minY[i], p.y);
                maxY[i] = Math.max(maxY[i], p.y);
            }
        }

        // 创建索引并按Y坐标排序
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, Comparator.comparingDouble(a -> minY[a]));

        // 计算平均高度作为行分组阈值
        double avgHeight = 0;
        for (int i = 0; i < n; i++) {
            avgHeight += (maxY[i] - minY[i]);
        }
        avgHeight /= n;
        double rowThreshold = avgHeight * 0.6;

        // 分组行
        List<List<Integer>> rows = new ArrayList<>();
        List<Integer> currentRow = new ArrayList<>();
        double currentRowY = minY[indices[0]];

        for (int idx : indices) {
            if (Math.abs(minY[idx] - currentRowY) <= rowThreshold) {
                currentRow.add(idx);
            } else {
                if (!currentRow.isEmpty()) {
                    rows.add(new ArrayList<>(currentRow));
                    currentRow.clear();
                }
                currentRow.add(idx);
                currentRowY = minY[idx];
            }
        }
        rows.add(currentRow);

        // 每行内按X坐标排序，构建结果
        int position = 0;
        for (List<Integer> row : rows) {
            row.sort(Comparator.comparingDouble(a -> minX[a]));
            for (int idx : row) {
                result.put(position++, points.get(idx));
            }
        }

        return result;
    }

    /**
     * 概率图转 Mat
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
            throw new IllegalStateException("failed to create directory: " + dir);
        }
    }

    public static void releaseMat(Mat mat) {
        if (mat != null) {
            mat.release();
        }
    }
}
