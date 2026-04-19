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
 * 完全兼容 PaddleOCR 官方实现
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
        int maxSideLen = config.getDetMaxSideLen();
        int align = 32;

        int originalWidth = context.getOriginalWidth();
        int originalHeight = context.getOriginalHeight();
        int maxOriginalSide = Math.max(originalWidth, originalHeight);

        // 1. 计算缩放比例（只缩小，不放大）
        float scale = 1.0f;
        if (maxOriginalSide > maxSideLen) {
            scale = (float) maxSideLen / maxOriginalSide;
        }

        // 2. 计算缩放后尺寸
        int newWidth = (int) (originalWidth * scale);
        int newHeight = (int) (originalHeight * scale);

        // 3. 对齐到32的倍数（向下取整）
        newWidth = newWidth - (newWidth % align);
        newHeight = newHeight - (newHeight % align);

        // 4. 确保最小尺寸（防止为0）
        newWidth = Math.max(newWidth, align);
        newHeight = Math.max(newHeight, align);

        // 5. 图像处理（保持 BGR 通道顺序，官方检测模型使用BGR）
        //    官方检测模型只做1/255归一化，不减均值不减标准差
        Mat processed = preprocess(context.getRawMat(), newWidth, newHeight);

        // 6. 记录预处理信息
        log.debug("预处理完成: 原始图像宽高 {}x{} -> 预处理后宽高 {}x{} (缩放比例: scale={}, 最大边长限制: maxSideLen={})",
                context.getOriginalWidth(), context.getOriginalHeight(), newWidth, newHeight,
                String.format("%.3f", scale), maxSideLen);

        context.setDetPrepWidth(newWidth);
        context.setDetPrepHeight(newHeight);
        context.setDetPrepMat(processed);
        context.setScale(scale);
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
        float dilationRatio = config.getDetDilationRatio();

        // 使用可配置的核大小
        int dilateKernelSize = Math.max(1, (int) dilationRatio);
        if (config.getDetDilateKernelSize() > 0) {
            dilateKernelSize = config.getDetDilateKernelSize();
        }

        int closeKernelSize = config.getDetCloseKernelSize();
        if (closeKernelSize <= 0) {
            closeKernelSize = 2;  // 默认值
        }

        return MatPipeline.fromMap(probMap)
                .binary(config.getDetDbThresh())
                .dilate(new Size(dilateKernelSize, dilateKernelSize))
                .close(new Size(closeKernelSize, closeKernelSize))
                .map(mat -> {
                    List<MatOfPoint> contours = findAndSortContours(mat);
                    log.debug("找到轮廓数量: {}", contours.size());
                    if (contours.isEmpty()) {
                        return new ArrayList<>();
                    }
                    List<List<Point>> allBoxes = extractTextBoxes(contours);
                    List<List<Point>> nmsBoxes = nms(allBoxes);
                    log.debug("NMS后剩余 {} 个文本框", nmsBoxes.size());
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

    /**
     * 从轮廓列表中提取文本框
     */
    private List<List<Point>> extractTextBoxes(List<MatOfPoint> contours) {
        List<List<Point>> allBoxes = new ArrayList<>();
        float unclipRatio = config.getDetDbUnclipRatio();
        int minBoxArea = config.getMinBoxArea() > 0 ? config.getMinBoxArea() : 3;

        for (MatOfPoint contour : contours) {
            // 使用修复后的 extractTextbox 方法，现在包含完整的 unclip 逻辑
            List<Point> box = OpenCVResource.extractTextbox(
                    contour,
                    unclipRatio,
                    minBoxArea
            );
            if (box != null && box.size() == 4) {
                // 确保点是顺时针顺序（官方要求）
                List<Point> orderedBox = OpenCVUtil.orderPoints(box);
                allBoxes.add(orderedBox);
            }
        }
        return allBoxes;
    }

  

    /**
     * NMS非极大值抑制
     */
    private List<List<Point>> nms(List<List<Point>> boxes) {
        if (boxes.size() < 2) {
            return boxes;
        }

        List<Rect> rects = new ArrayList<>();
        List<Double> areas = new ArrayList<>();
        for (List<Point> box : boxes) {
            Rect rect = OpenCVUtil.getBoundingRect(box);
            rects.add(rect);
            areas.add(rect.area());
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Double.compare(areas.get(b), areas.get(a)));

        boolean[] suppressed = new boolean[boxes.size()];
        List<List<Point>> result = new ArrayList<>();

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
                if (iou > DEFAULT_NMS_IOU_THRESHOLD) {
                    suppressed[otherIdx] = true;
                }
            }
        }

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