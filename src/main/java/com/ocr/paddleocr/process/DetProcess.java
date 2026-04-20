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
import org.opencv.imgproc.Imgproc;

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

    // ==================== 后处理默认参数 ====================
    private static final float DEFAULT_NMS_IOU_THRESHOLD = 0.5f;
    private static final int DEFAULT_MIN_BOX_AREA = 9;

    // ==================== 框过滤参数 ====================
    private static final int DEFAULT_MIN_BOX_WIDTH = 5;
    private static final int DEFAULT_MIN_BOX_HEIGHT = 5;
    private static final float DEFAULT_MAX_ASPECT_RATIO = 50.0f;

    // ==================== 成员变量 ====================
    private OrtSession session;
    private final OrtEnvironment env;
    private final OCRConfig config;

    public DetProcess(OCRConfig config) throws OrtException {
        this.config = config;
        this.env = OrtEnvironment.getEnvironment();
        loadModel();
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
            log.debug("分割图尺寸: {}x{}", parseResult[0].length, parseResult[0][0].length);

            // 5. DB后处理获取文本框（传入概率图用于置信度计算）
            List<List<Point>> boxes = dbPostProcess(parseResult[0], context);
            log.info("DB后处理检测到 {} 个文本框", boxes.size());

            // 6. 过滤无效框
            boxes = filterInvalidBoxes(boxes, context.getOriginalWidth(), context.getOriginalHeight());
            log.info("过滤后剩余 {} 个文本框", boxes.size());

            // 7. 设置识别结果
            context.setBoxes(boxes.stream()
                    .map(box -> {
                        TextBox textBox = new TextBox();
                        textBox.setBox(box);
                        return textBox;
                    }).collect(Collectors.toList()));

            // 8. 释放资源
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
     * DB 检测模型预处理（完全遵循 PaddleOCR 官方策略）
     */
    private void detPreprocess(ModelProcessContext context) {
        // 模型要求的固定输入尺寸（从配置中读取）
        int modelInputHeight = config.getDetImageHeight();  // 应该是 960
        int modelInputWidth = config.getDetImageWidth();    // 应该是 960

        int originalWidth = context.getOriginalWidth();
        int originalHeight = context.getOriginalHeight();

        // 计算缩放比例（保持长宽比，缩放到模型输入尺寸内）
        float scale = Math.min(
                (float) modelInputWidth / originalWidth,
                (float) modelInputHeight / originalHeight
        );

        // 计算缩放后的尺寸
        int newWidth = (int) (originalWidth * scale);
        int newHeight = (int) (originalHeight * scale);

        // 确保尺寸不小于1
        newWidth = Math.max(newWidth, 1);
        newHeight = Math.max(newHeight, 1);

        // 创建缩放后的图像
        Mat resized = new Mat();
        Imgproc.resize(context.getRawMat(), resized, new Size(newWidth, newHeight));

        // 创建目标尺寸的空白图像（黑色填充）
        Mat padded = new Mat(modelInputHeight, modelInputWidth, resized.type(), new Scalar(0, 0, 0));

        // 将缩放后的图像放到中心位置
        int xOffset = (modelInputWidth - newWidth) / 2;
        int yOffset = (modelInputHeight - newHeight) / 2;
        resized.copyTo(padded.submat(yOffset, yOffset + newHeight, xOffset, xOffset + newWidth));

        // 归一化到 [0, 1] 范围（检测模型需要）
        Mat floatMat = new Mat();
        padded.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);

        // 释放临时资源
        resized.release();
        padded.release();

        // 保存预处理结果
        context.setDetPrepWidth(modelInputWidth);
        context.setDetPrepHeight(modelInputHeight);
        context.setDetPrepMat(floatMat);

        // 保存缩放比例和偏移量（用于后续坐标转换）
        context.setScale(scale);
//        context.setDetPadTop(yOffset);
//        context.setDetPadLeft(xOffset);

        log.debug("预处理完成: 原始 {}x{} -> 缩放 {}x{} -> 填充 {}x{}, 缩放比例={}",
                originalWidth, originalHeight, newWidth, newHeight,
                modelInputWidth, modelInputHeight, String.format("%.3f", scale));
    }

    /**
     * 预处理：保持 BGR 通道顺序
     */
    private Mat preprocess(Mat image, int targetWidth, int targetHeight) {
        return MatPipeline.fromMat(image)
                .resize(targetWidth, targetHeight)
                .normalize()
                .get();
    }

    /**
     * 解析模型输出
     */
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
     * DB后处理算法
     */
    private List<List<Point>> dbPostProcess(float[][] probMap, ModelProcessContext context) {
        float boxThresh = config.getDetDbBoxThresh();
        float unclipRatio = config.getDetDbUnclipRatio();
        int minBoxArea = config.getMinBoxArea() > 0 ? config.getMinBoxArea() : 10;

        int height = probMap.length;
        int width = probMap[0].length;

        // 添加：打印概率图统计信息
        float minProb = Float.MAX_VALUE;
        float maxProb = Float.MIN_VALUE;
        float sumProb = 0;
        for (float[] floats : probMap) {
            for (int j = 0; j < width; j++) {
                float val = floats[j];
                minProb = Math.min(minProb, val);
                maxProb = Math.max(maxProb, val);
                sumProb += val;
            }
        }
        float meanProb = sumProb / (height * width);
        log.info("概率图统计: 尺寸={}x{}, 范围=[{}, {}], 均值={}",
                height, width, minProb, maxProb, meanProb);
        log.info("当前阈值: boxThresh={}, unclipRatio={}, minBoxArea={}",
                boxThresh, unclipRatio, minBoxArea);

        return MatPipeline.fromMap(probMap)
                .peek(mat -> {
                    // 打印二值化前的统计
                    log.debug("概率图类型: {}", mat.type());
                })
                .binary(boxThresh)
                .peek(mat -> {
                    // 打印二值化后的白点数量
                    int whiteCount = Core.countNonZero(mat);
                    log.info("二值化后白点数量: {} / {} ({}%)",
                            whiteCount, height * width, 100.0 * whiteCount / (height * width));
                })
                .toCV8UC1()
                .peek(mat -> {
                    int whiteCount = Core.countNonZero(mat);
                    log.info("转换CV8UC1后白点数量: {}", whiteCount);
                })
                .dilate(new Size(config.getDetDilateKernelSize(), config.getDetDilateKernelSize()))
                .peek(mat -> {
                    int whiteCount = Core.countNonZero(mat);
                    log.info("膨胀后白点数量: {}", whiteCount);
                })
                .close(new Size(config.getDetCloseKernelSize(), config.getDetCloseKernelSize()))
                .peek(mat -> {
                    int whiteCount = Core.countNonZero(mat);
                    log.info("闭运算后白点数量: {}", whiteCount);
                })
                .map(mat -> {
                    List<MatOfPoint> contours = findAndSortContours(mat);
                    log.info("找到轮廓数量: {}", contours.size());

                    if (contours.isEmpty()) {
                        return new ArrayList<>();
                    }

                    List<List<Point>> allBoxes = extractTextBoxes(contours);
                    log.info("提取文本框数量: {}", allBoxes.size());

                    List<List<Point>> nmsBoxes = nms(allBoxes);
                    log.info("NMS后文本框数量: {}", nmsBoxes.size());

                    return restoreBoxesToOriginal(nmsBoxes, context);
                });
    }
    /**
     * 查找并排序轮廓（官方按面积降序）
     */
    private List<MatOfPoint> findAndSortContours(Mat binary) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();

        // 按轮廓面积降序排序（官方做法）
        contours.sort((a, b) -> Double.compare(
                Imgproc.contourArea(b), Imgproc.contourArea(a)));

        return contours;
    }

    private List<List<Point>> extractTextBoxes(List<MatOfPoint> contours) {
        List<List<Point>> allBoxes = new ArrayList<>();
        float unclipRatio = config.getDetDbUnclipRatio();
        int minBoxArea = config.getMinBoxArea() > 0 ? config.getMinBoxArea() : 3;  // 降低到 3

        int skippedByArea = 0;
        int skippedByPoints = 0;
        int skippedByBox = 0;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < minBoxArea) {
                skippedByArea++;
                continue;
            }

            // 检查轮廓点数
            Point[] points = contour.toArray();
            if (points.length < 4) {  // 降低到 4
                skippedByPoints++;
                continue;
            }

            log.debug("轮廓面积: {}, 点数: {}", area, points.length);

            try {
                List<Point> box = OpenCVResource.extractTextbox(
                        contour,
                        unclipRatio,
                        minBoxArea
                );

                if (box != null && box.size() == 4) {
                    List<Point> orderedBox = OpenCVUtil.orderPoints(box);
                    allBoxes.add(orderedBox);
                    log.debug("成功提取文本框: 面积={}", area);
                } else {
                    skippedByBox++;
                    log.debug("extractTextbox 返回 null 或点数不对: {}", box != null ? box.size() : "null");
                }
            } catch (Exception e) {
                log.debug("extractTextbox 异常: {}", e.getMessage());
                skippedByBox++;
            }
        }

        log.info("文本框提取统计: 总面积过滤={}, 点数不足={}, 提取失败={}, 成功={}",
                skippedByArea, skippedByPoints, skippedByBox, allBoxes.size());

        return allBoxes;
    }



    /**
     * NMS非极大值抑制
     */
    private List<List<Point>> nms(List<List<Point>> boxes) {
        float iouThreshold = config.getNmsIouThreshold();

        if (boxes.size() < 2) {
            log.info("NMS: 文本框数量少于2，跳过");
            return boxes;
        }

        log.info("NMS开始: 输入={}, IoU阈值={}", boxes.size(), iouThreshold);

        List<Rect> rects = new ArrayList<>();
        List<Double> areas = new ArrayList<>();
        for (List<Point> box : boxes) {
            Rect rect = OpenCVUtil.getBoundingRect(box);
            rects.add(rect);
            areas.add(rect.area());
            log.info("文本框: rect={}, area={}", rect, rect.area());
        }

        // 按面积降序排序
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Double.compare(areas.get(b), areas.get(a)));

        log.info("NMS排序后前5个面积: {}",
                indices.stream().limit(5).map(areas::get).collect(Collectors.toList()));

        boolean[] suppressed = new boolean[boxes.size()];
        List<List<Point>> result = new ArrayList<>();
        int suppressedCount = 0;

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            if (suppressed[idx]) {
                continue;
            }

            result.add(boxes.get(idx));

            for (int j = i + 1; j < indices.size(); j++) {
                int otherIdx = indices.get(j);
                if (suppressed[otherIdx]) {
                    continue;
                }

                float iou = OpenCVUtil.computeIoU(rects.get(idx), rects.get(otherIdx));

                if (iou > iouThreshold) {
                    suppressed[otherIdx] = true;
                    suppressedCount++;
                    log.trace("抑制文本框: iou={}, idx={}", iou, otherIdx);
                }
            }
        }

        log.info("NMS完成: 输入={}, 输出={}, 抑制={}, IoU阈值={}",
                boxes.size(), result.size(), suppressedCount, iouThreshold);

        return result;
    }

    /**
     * 还原文本框到原始图像尺寸
     */
    private List<List<Point>> restoreBoxesToOriginal(List<List<Point>> boxes,
                                                     ModelProcessContext context) {
        float scale = context.getScale();
        int originalWidth = context.getOriginalWidth();
        int originalHeight = context.getOriginalHeight();

        return boxes.stream()
                .map(box -> box.stream()
                        .map(p -> new Point(
                                clamp(p.x / scale, originalWidth - 1),
                                clamp(p.y / scale, originalHeight - 1)
                        ))
                        .collect(Collectors.toList())
                )
                .collect(Collectors.toList());
    }

    private double clamp(double value, double max) {
        return Math.min(Math.max(value, 0), max);
    }

    /**
     * 过滤无效检测框
     */
    private List<List<Point>> filterInvalidBoxes(List<List<Point>> boxes, int imgWidth, int imgHeight) {
        if (boxes.isEmpty()) {
            return boxes;
        }

        List<List<Point>> filtered = new ArrayList<>();

        for (List<Point> box : boxes) {
            if (box.size() != 4) {
                continue;
            }

            // 边界检查
            boolean outOfBounds = false;
            for (Point p : box) {
                if (p.x < -5 || p.x > imgWidth + 5 || p.y < -5 || p.y > imgHeight + 5) {
                    outOfBounds = true;
                    break;
                }
            }
            if (outOfBounds) {
                continue;
            }

            Rect rect = OpenCVUtil.getBoundingRect(box);

            // 尺寸检查
            if (rect.width < DEFAULT_MIN_BOX_WIDTH || rect.height < DEFAULT_MIN_BOX_HEIGHT) {
                log.debug("过滤：框太小 ({}x{})", rect.width, rect.height);
                continue;
            }

            // 宽高比检查
            float aspectRatio = (float) Math.max(rect.width, rect.height) / Math.min(rect.width, rect.height);
            if (aspectRatio > DEFAULT_MAX_ASPECT_RATIO) {
                log.debug("过滤：宽高比过大 ({})", aspectRatio);
                continue;
            }

            // 面积检查
            if (rect.area() < DEFAULT_MIN_BOX_AREA) {
                log.debug("过滤：面积过小 ({})", rect.area());
                continue;
            }

            filtered.add(box);
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