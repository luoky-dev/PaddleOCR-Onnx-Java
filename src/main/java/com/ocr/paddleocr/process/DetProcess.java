package com.ocr.paddleocr.process;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ModelProcessContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.*;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 文本检测处理器
 */
@Slf4j
public class DetProcess implements AutoCloseable {

    // ==================== 后处理默认参数（对齐官方） ====================
    private static final float DEFAULT_NMS_IOU_THRESHOLD = 0.3f;  // 官方默认 0.3
    private static final int DEFAULT_MIN_BOX_AREA = 10;           // 官方默认 10

    // ==================== 成员变量 ====================
    private OrtSession session;
    private final OrtEnvironment env;
    private final OCRConfig config;
    private final boolean debugMode;

    // 中间图片保存目录
    private String intermediateDir;

    public DetProcess(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        this.debugMode = config.isDebugMode();
        this.intermediateDir =  "src/main/java/resources/test/output";
        loadModel();

        // 创建中间图片保存目录
        if (config.isVisualize()) {
            File dir = new File(intermediateDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            log.info("中间图片保存目录: {}", intermediateDir);
        }

        log.info("文本检测处理器初始化完成，模型路径: {}", config.getDetModelPath());
    }

    private void loadModel() throws OrtException {
        SessionOptions sessionOptions = ModelUtil.getSessionOptions(config);
        session = env.createSession(config.getDetModelPath(), sessionOptions);
        log.info("检测模型加载成功: {}", config.getDetModelPath());
    }

    /**
     * 执行文本检测
     */
    public void detect(ModelProcessContext context) {
        long startTime = System.currentTimeMillis();
        try {
            // 1. 预处理
            detPreprocess(context);

            // 2. 创建输入Tensor
            OnnxTensor inputTensor = OnnxUtil.createInputTensor(context.getDetPrepMat(), env);

            // 3. 执行推理
            Map<String, OnnxTensor> inputs = Collections.singletonMap("x", inputTensor);
            Result output = session.run(inputs);

            // 4. 解析输出
            float[][][] parseResult = parseOutput(output);
            log.info("分割图尺寸: {}x{}", parseResult[0].length, parseResult[0][0].length);

            // 保存概率图热力图
            if (config.isVisualize()) {
                saveProbabilityHeatmap(parseResult[0], "prob_map");
            }

            // 5. DB后处理获取文本框
            List<List<Point>> boxes = dbPostProcess(parseResult[0], context);
            log.info("DB后处理检测到 {} 个文本框", boxes.size());

            // 6. 绘制并保存检测框图
            if (config.isVisualize()) {
                Mat originalImage = context.getRawMat();
                Mat visualized = drawDetectionBoxes(originalImage, boxes);
                saveImage(visualized, "detection_boxes");
                visualized.release();

                // 保存边缘检测图
                saveEdgeMap(originalImage, "edges");
            }

            // 7. 过滤无效框（对齐官方）
            boxes = filterInvalidBoxes(boxes, context.getOriginalWidth(), context.getOriginalHeight());
            log.info("过滤后剩余 {} 个文本框", boxes.size());

            // 8. 设置识别结果
            context.setBoxes(boxes.stream()
                    .map(box -> {
                        TextBox textBox = new TextBox();
                        textBox.setBoxPoint(box);
                        return textBox;
                    }).collect(Collectors.toList()));

            // 9. 释放资源
            inputTensor.close();
            output.close();

            context.setSuccess(true);
            context.setDetProcessTime(System.currentTimeMillis() - startTime);

            log.info("文本检测完成，耗时: {}ms", context.getDetProcessTime());

        } catch (Exception e) {
            context.setSuccess(false);
            context.setError(e.getMessage());
            log.error("文本检测失败", e);
        }
    }

    /**
     * 保存概率图热力图
     */
    private void saveProbabilityHeatmap(float[][] probMap, String filename) {
        try {
            int height = probMap.length;
            int width = probMap[0].length;

            // 创建概率图Mat
            Mat probMat = new Mat(height, width, CvType.CV_32FC1);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    probMat.put(i, j, probMap[i][j] * 255);
                }
            }

            // 转换为8位图像
            Mat prob8u = new Mat();
            probMat.convertTo(prob8u, CvType.CV_8UC1);

            // 应用彩色映射
            Mat heatmap = new Mat();
            Imgproc.applyColorMap(prob8u, heatmap, Imgproc.COLORMAP_JET);

            // 保存
            saveImage(heatmap, filename + "_heatmap");

            // 保存原始概率图（灰度）
            saveImage(prob8u, filename + "_gray");

            probMat.release();
            prob8u.release();
            heatmap.release();

            log.debug("保存概率图: {}", filename);
        } catch (Exception e) {
            log.warn("保存概率图失败: {}", e.getMessage());
        }
    }


    /**
     * 保存边缘检测图
     */
    private void saveEdgeMap(Mat image, String filename) {
        try {
            Mat gray = new Mat();
            Mat edges = new Mat();
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

            // Canny边缘检测
            Imgproc.Canny(gray, edges, 50, 150);
            saveImage(edges, filename + "_canny");

            gray.release();
            edges.release();
            log.debug("保存边缘检测图: {}", filename);
        } catch (Exception e) {
            log.warn("保存边缘检测图失败: {}", e.getMessage());
        }
    }

    /**
     * 绘制检测框
     */
    private Mat drawDetectionBoxes(Mat image, List<List<Point>> boxes) {
        Mat result = image.clone();
        Scalar color = new Scalar(0, 255, 0);  // 绿色
        int thickness = 2;

        for (int i = 0; i < boxes.size(); i++) {
            List<Point> box = boxes.get(i);
            if (box == null || box.size() < 4) {
                continue;
            }

            MatOfPoint matOfPoint = new MatOfPoint();
            matOfPoint.fromList(box);
            Imgproc.polylines(result, Collections.singletonList(matOfPoint), true, color, thickness);
            matOfPoint.release();

            // 绘制序号
            Point center = getBoxCenter(box);
            Imgproc.putText(result, String.valueOf(i + 1),
                    new org.opencv.core.Point(center.x, center.y - 5),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        return result;
    }

    /**
     * 获取框的中心点
     */
    private Point getBoxCenter(List<Point> box) {
        double sumX = 0, sumY = 0;
        for (Point p : box) {
            sumX += p.x;
            sumY += p.y;
        }
        return new Point(sumX / box.size(), sumY / box.size());
    }

    /**
     * 保存图像到文件
     */
    private void saveImage(Mat image, String filename) {
        try {
            String path = intermediateDir + "/" + filename + ".jpg";
            Imgcodecs.imwrite(path, image);
            log.debug("保存图片: {}", path);
        } catch (Exception e) {
            log.warn("保存图片失败: {} - {}", filename, e.getMessage());
        }
    }

    /**
     * 保存二值化图像（用于调试）
     */
    private void saveBinaryImage(Mat binary, String filename) {
        if (!config.isVisualize()) return;

        try {
            // 确保是8位单通道
            Mat binary8u = new Mat();
            if (binary.type() == CvType.CV_32FC1) {
                binary.convertTo(binary8u, CvType.CV_8UC1, 255);
            } else {
                binary8u = binary.clone();
            }
            saveImage(binary8u, filename);
            binary8u.release();
        } catch (Exception e) {
            log.warn("保存二值化图失败: {}", e.getMessage());
        }
    }

    /**
     * 保存膨胀/闭运算后的图像
     */
    private void saveMorphImage(Mat morph, String filename) {
        if (!config.isVisualize()) return;
        saveImage(morph, filename);
    }

    /**
     * 保存轮廓图像
     */
    private void saveContoursImage(Mat binary, List<MatOfPoint> contours, String filename) {
        if (!config.isVisualize()) return;

        try {
            Mat contoursImg = new Mat();
            Imgproc.cvtColor(binary, contoursImg, Imgproc.COLOR_GRAY2BGR);

            // 绘制轮廓
            for (int i = 0; i < contours.size(); i++) {
                Scalar color = new Scalar(
                        (i * 50) % 255,
                        (i * 100) % 255,
                        (i * 150) % 255
                );
                Imgproc.drawContours(contoursImg, contours, i, color, 1);
            }

            saveImage(contoursImg, filename);
            contoursImg.release();
        } catch (Exception e) {
            log.warn("保存轮廓图失败: {}", e.getMessage());
        }
    }

    // ==================== 原有的预处理和后处理方法 ====================

    /**
     * DB 检测模型预处理
     */
    private void detPreprocess(ModelProcessContext context) {
        int maxSideLen = config.getDetMaxSideLen();
        int align = 32;

        int originalWidth = context.getOriginalWidth();
        int originalHeight = context.getOriginalHeight();

        float scale;
        int maxOriginalSide = Math.max(originalWidth, originalHeight);
        if (maxOriginalSide > maxSideLen) {
            scale = (float) maxSideLen / maxOriginalSide;
        } else {
            scale = 1.0f;
        }

        int newWidth = (int) (originalWidth * scale);
        int newHeight = (int) (originalHeight * scale);

        newWidth = newWidth - (newWidth % align);
        newHeight = newHeight - (newHeight % align);

        newWidth = Math.max(newWidth, align);
        newHeight = Math.max(newHeight, align);

        Mat resized = new Mat();
        Imgproc.resize(context.getRawMat(), resized, new Size(newWidth, newHeight));

        // 保存缩放后的图像
        if (config.isVisualize()) {
            saveImage(resized, "preprocessed_resized");
        }

        Mat floatMat = new Mat();
        resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);
        resized.release();

        Mat normalized = normalizeImage(floatMat);
        floatMat.release();

        context.setDetPrepWidth(newWidth);
        context.setDetPrepHeight(newHeight);
        context.setDetPrepMat(normalized);
        context.setScale(scale);

        log.debug("预处理完成: 原始 {}x{} -> 缩放 {}x{} (scale={}, 对齐32)",
                originalWidth, originalHeight, newWidth, newHeight, scale);
    }

    /**
     * 图像标准化（减均值，除标准差）
     */
    private Mat normalizeImage(Mat image) {
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        Mat result = new Mat();
        image.convertTo(result, CvType.CV_32FC3);

        List<Mat> channels = new ArrayList<>();
        Core.split(result, channels);

        for (int i = 0; i < 3; i++) {
            Mat channel = channels.get(i);
            Core.subtract(channel, new Scalar(mean[i]), channel);
            Core.divide(channel, new Scalar(std[i]), channel);
        }

        Core.merge(channels, result);

        for (Mat ch : channels) {
            ch.release();
        }

        return result;
    }

    private float[][][] parseOutput(Result output) throws OrtException {
        OnnxValue outputValue = output.get(0);
        try {
            float[][][][] outputData4D = (float[][][][]) outputValue.getValue();
            return new float[][][]{outputData4D[0][0]};
        } catch (ClassCastException e1) {
            try {
                return (float[][][]) outputValue.getValue();
            } catch (ClassCastException e2) {
                throw new OrtException("无法解析检测模型输出格式");
            }
        }
    }

    /**
     * DB后处理 - 完全对齐官方
     */
    /**
     * DB后处理 - 完全对齐官方
     */
    private List<List<Point>> dbPostProcess(float[][] probMap, ModelProcessContext context) {
        // 1. 二值化
        boolean[][] bitmap = new boolean[probMap.length][probMap[0].length];
        for (int i = 0; i < probMap.length; i++) {
            for (int j = 0; j < probMap[0].length; j++) {
                bitmap[i][j] = probMap[i][j] > DBPostProcess.THRESH;
            }
        }

        // 2. 膨胀操作
        bitmap = DBPostProcess.expandBitmap(bitmap, probMap, DBPostProcess.BOX_THRESH);

        // 3. 查找轮廓
        List<MatOfPoint> contours = DBPostProcess.findContours(bitmap);

        // 保存轮廓图像
        if (config.isVisualize() && !contours.isEmpty()) {
            Mat binaryMat = new Mat(probMap.length, probMap[0].length, CvType.CV_8UC1);
            for (int i = 0; i < probMap.length; i++) {
                for (int j = 0; j < probMap[0].length; j++) {
                    binaryMat.put(i, j, bitmap[i][j] ? 255 : 0);
                }
            }
            saveContoursImage(binaryMat, contours, "contours");
            binaryMat.release();
        }

        // 4. 提取文本框
        List<List<Point>> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < DBPostProcess.MIN_AREA) continue;

            double score = DBPostProcess.getScore(contour, probMap);
            if (score < DBPostProcess.BOX_THRESH) continue;

            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            RotatedRect rect = Imgproc.minAreaRect(contour2f);
            contour2f.release();

            double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            double distance = area * DBPostProcess.UNCLIP_RATIO / perimeter;

            List<Point> box = DBPostProcess.unclip(rect, distance);
            double epsilon = 0.002 * Imgproc.arcLength(new MatOfPoint2f(box.toArray(new Point[0])), true);
            box = DBPostProcess.approxPolyDP(box, epsilon, true);

            if (box.size() == 4) {
                boxes.add(box);
                scores.add((float) score);
            }
        }

        log.info("NMS前文本框数量: {}", boxes.size());

        // 打印NMS前的坐标信息
        for (int i = 0; i < Math.min(boxes.size(), 5); i++) {
            List<Point> box = boxes.get(i);
            log.info("NMS前[{}]: 坐标点:", i);
            for (int j = 0; j < box.size(); j++) {
                Point p = box.get(j);
                log.info("  点{}: ({}, {})", j, p.x, p.y);
            }
            // 计算边界框
            double minX = box.stream().mapToDouble(p -> p.x).min().orElse(0);
            double minY = box.stream().mapToDouble(p -> p.y).min().orElse(0);
            double maxX = box.stream().mapToDouble(p -> p.x).max().orElse(0);
            double maxY = box.stream().mapToDouble(p -> p.y).max().orElse(0);
            log.info("NMS前[{}] 边界框: x=[{}, {}], y=[{}, {}], 宽={}, 高={}",
                    i, minX, maxX, minY, maxY, maxX - minX, maxY - minY);
        }

        // 5. NMS
        List<Integer> indices = DBPostProcess.nmsBoxes(boxes, scores, DBPostProcess.BOX_THRESH, DBPostProcess.NMS_IOU_THRESH);
        List<List<Point>> nmsBoxes = new ArrayList<>();
        for (int idx : indices) {
            nmsBoxes.add(boxes.get(idx));
        }

        log.info("NMS后文本框数量: {}", nmsBoxes.size());

        // 打印NMS后的坐标信息
        for (int i = 0; i < Math.min(nmsBoxes.size(), 5); i++) {
            List<Point> box = nmsBoxes.get(i);
            log.info("NMS后[{}]: 坐标点:", i);
            for (int j = 0; j < box.size(); j++) {
                Point p = box.get(j);
                log.info("  点{}: ({}, {})", j, p.x, p.y);
            }
            double minX = box.stream().mapToDouble(p -> p.x).min().orElse(0);
            double minY = box.stream().mapToDouble(p -> p.y).min().orElse(0);
            double maxX = box.stream().mapToDouble(p -> p.x).max().orElse(0);
            double maxY = box.stream().mapToDouble(p -> p.y).max().orElse(0);
            log.info("NMS后[{}] 边界框: x=[{}, {}], y=[{}, {}], 宽={}, 高={}",
                    i, minX, maxX, minY, maxY, maxX - minX, maxY - minY);
        }

        // 6. 坐标还原
        float scale = context.getScale();
        log.info("========== 坐标还原 ==========");
        log.info("缩放比例 scale: {}", scale);
        log.info("原始图像尺寸: {}x{}", context.getOriginalWidth(), context.getOriginalHeight());
        log.info("还原前文本框数量: {}", nmsBoxes.size());

        List<List<Point>> restoredBoxes = DBPostProcess.restoreBoxes(nmsBoxes, scale);

        log.info("还原后文本框数量: {}", restoredBoxes.size());

        // 打印还原后的坐标信息
        for (int i = 0; i < Math.min(restoredBoxes.size(), 5); i++) {
            List<Point> box = restoredBoxes.get(i);
            log.info("还原后[{}]: 坐标点:", i);
            for (int j = 0; j < box.size(); j++) {
                Point p = box.get(j);
                log.info("  点{}: ({}, {})", j, p.x, p.y);
            }
            double minX = box.stream().mapToDouble(p -> p.x).min().orElse(0);
            double minY = box.stream().mapToDouble(p -> p.y).min().orElse(0);
            double maxX = box.stream().mapToDouble(p -> p.x).max().orElse(0);
            double maxY = box.stream().mapToDouble(p -> p.y).max().orElse(0);
            log.info("还原后[{}] 边界框: x=[{}, {}], y=[{}, {}], 宽={}, 高={}",
                    i, minX, maxX, minY, maxY, maxX - minX, maxY - minY);

            // 验证坐标是否在原始图像范围内
            boolean inBounds = true;
            for (Point p : box) {
                if (p.x < 0 || p.x > context.getOriginalWidth() ||
                        p.y < 0 || p.y > context.getOriginalHeight()) {
                    inBounds = false;
                    break;
                }
            }
            log.info("还原后[{}] 坐标是否在图像范围内: {}", i, inBounds);
        }

        log.info("坐标还原完成");

        return restoredBoxes;
    }
    /**
     * 过滤无效检测框（对齐官方）
     */
    private List<List<Point>> filterInvalidBoxes(List<List<Point>> boxes, int imgWidth, int imgHeight) {
        if (boxes.isEmpty()) {
            return boxes;
        }

        List<List<Point>> filtered = new ArrayList<>();
        int outOfBoundsCount = 0;

        for (List<Point> box : boxes) {
            if (box.size() != 4) {
                continue;
            }

            boolean outOfBounds = false;
            for (Point p : box) {
                if (p.x < -5 || p.x > imgWidth + 5 || p.y < -5 || p.y > imgHeight + 5) {
                    outOfBounds = true;
                    break;
                }
            }
            if (outOfBounds) {
                outOfBoundsCount++;
                continue;
            }

            Rect rect = OpenCVUtil.getBoundingRect(box);
            if (rect.area() < DEFAULT_MIN_BOX_AREA) {
                log.debug("过滤：面积过小 ({})", rect.area());
                continue;
            }

            filtered.add(box);
        }

        if (outOfBoundsCount > 0) {
            log.debug("过滤：{} 个文本框超出边界", outOfBoundsCount);
        }

        log.debug("框过滤完成: {} -> {}", boxes.size(), filtered.size());
        return filtered;
    }

    @Override
    public void close() throws OrtException {
        if (session != null) {
            session.close();
            log.info("检测模型会话已关闭");
        }
        if (env != null) {
            env.close();
        }
    }
}