package com.ocr.paddleocr.utils;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.List;

@Slf4j
public class ImageUtil {

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
            log.warn("图像为空，无法绘制");
            return null;
        }

        if (boxes == null || boxes.isEmpty()) {
            log.warn("检测框列表为空，无法绘制");
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
     * 在图像上绘制检测框（带置信度）
     *
     * @param image 原始图像
     * @param boxes 检测框坐标列表
     * @param confidences 置信度列表
     * @param thickness 线条粗细
     * @return 绘制后的图像
     */
    public static Mat drawBoxesWithConfidence(Mat image, List<List<Point>> boxes,
                                              List<Float> confidences, int thickness) {
        if (image == null || image.empty()) {
            log.warn("图像为空，无法绘制");
            return null;
        }

        Mat result = image.clone();

        for (int i = 0; i < boxes.size(); i++) {
            List<Point> box = boxes.get(i);
            if (box == null || box.size() < 4) {
                continue;
            }

            // 根据置信度选择颜色
            Scalar color = getColorByConfidence(confidences.get(i));

            // 绘制多边形
            drawPolygon(result, box, color, thickness);

            // 绘制置信度文本
            Point center = getCenter(box);
            String text = String.format("%.2f", confidences.get(i));
            putText(result, text, center, color);
        }

        return result;
    }

    /**
     * 绘制多边形
     */
    private static void drawPolygon(Mat image, List<Point> points, Scalar color, int thickness) {
        if (points == null || points.size() < 3) {
            return;
        }

        // 将点转换为 MatOfPoint
        MatOfPoint matOfPoint = new MatOfPoint();
        matOfPoint.fromList(points);

        // 绘制多边形轮廓
        Imgproc.polylines(image, java.util.Collections.singletonList(matOfPoint),
                true, color, thickness);

        matOfPoint.release();
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
    public static void saveDrawResult(Mat image, String outputPath) {
        if (image == null || image.empty()) {
            log.warn("图像为空，无法保存");
            return;
        }

        File outputFile = new File(outputPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        Imgcodecs.imwrite(outputPath, image);
        log.info("绘制结果已保存: {}", outputPath);
    }
}
