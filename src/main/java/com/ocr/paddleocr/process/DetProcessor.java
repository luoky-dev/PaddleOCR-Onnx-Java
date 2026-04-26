package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.OnnxUtil;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Slf4j
public class DetProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;

    public DetProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
        log.debug("DetProcessor初始化完成, 检测模型路径: {}, 最大边长: {}, 对齐步长: {}",
                ocrConfig.getDetModelPath(),
                modelConfig.getDetMaxSideLen(),
                modelConfig.getResizeAlign());
    }

    public void detect(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        // 预处理
        preprocess(context);
        // 模型解析
        parse(context);
        // 后处理
        postprocess(context);
        // 计算运行时间
        context.setDetProcessTime(System.currentTimeMillis() - startTime);
    }

    private void preprocess(OCRContext context) {
        long startTime = System.currentTimeMillis();
        // 获取原始图像尺寸
        Mat raw = context.getRawMat();
        int srcW = raw.cols();
        int srcH = raw.rows();
        int dstW;
        int dstH;
        log.debug("原始图像尺寸: {}x{}", srcW, srcH);
        if (!OnnxUtil.isDynamicInput(modelManager.getDetSession())) {
            // 固定输入模型: 严格按模型声明尺寸送入
            dstH = ocrConfig.getDetModelHeight();
            dstW = ocrConfig.getDetModelWidth();
            log.debug("固定尺寸模型, 目标尺寸: {}x{}", dstW, dstH);
        } else {
            // 动态输入模型: 沿用原有长边限制 + 对齐策略
            int maxSide = Math.max(srcW, srcH);
            float scale = maxSide > modelConfig.getDetMaxSideLen() ? (float) modelConfig.getDetMaxSideLen() / maxSide : 1.0f;
            dstW = Math.max((int) (srcW * scale), modelConfig.getResizeAlign());
            dstH = Math.max((int) (srcH * scale), modelConfig.getResizeAlign());
            dstW = Math.max((dstW / modelConfig.getResizeAlign()) * modelConfig.getResizeAlign(), modelConfig.getResizeAlign());
            dstH = Math.max((dstH / modelConfig.getResizeAlign()) * modelConfig.getResizeAlign(), modelConfig.getResizeAlign());
            log.debug("动态尺寸模型, 缩放比例: {}, 目标尺寸: {}x{}", scale, dstW, dstH);
        }
        // 缩放图像，使用双线性插值，保持图像内容不变形
        Mat resized = new Mat();
        Imgproc.resize(raw, resized, new Size(dstW, dstH));
        log.debug("图像缩放完成: {}x{} -> {}x{}", srcW, srcH, dstW, dstH);
        // 官方检测模型输入要求 RGB 顺序
        Mat rgb = new Mat();
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        // 归一化 + 标准化
        Mat normalized = OpenCVUtil.normalize(rgb,modelConfig.getMean(),modelConfig.getStd());
        OpenCVUtil.releaseMat(resized);
        OpenCVUtil.releaseMat(rgb);
        log.debug("图像归一化完成, 均值: {}, 标准差: {}",
                Arrays.toString(modelConfig.getMean()),
                Arrays.toString(modelConfig.getStd()));
        // 对齐后的真实缩放比例（用于坐标精确还原）
        float scaleX = srcW > 0 ? (float) dstW / (float) srcW : 1.0f;
        float scaleY = srcH > 0 ? (float) dstH / (float) srcH : 1.0f;
        // 保存结果
        context.setDetPrepMat(normalized);
        context.setDetPrepScaleX(scaleX);
        context.setDetPrepScaleY(scaleY);

        long elapsed = System.currentTimeMillis() - startTime;
        log.debug("检测预处理完成, 缩放比例: ({}, {}), 耗时: {}ms",
                scaleX, scaleY, elapsed);
    }

    /**
     * 模型推理
     */
    private void parse(OCRContext context) throws OrtException {
        long startTime = System.currentTimeMillis();
        log.debug("开始检测模型推理");

        // 模型输出
        try (OnnxTensor input = OnnxUtil.createInputTensor(context.getDetPrepMat(), modelManager.getEnv());
             Result output = modelManager.getDetSession().run(Collections.singletonMap("x", input))) {

            log.debug("模型推理完成, 输入形状: {}x{}x{}",
                    context.getDetPrepMat().cols(),
                    context.getDetPrepMat().rows(),
                    context.getDetPrepMat().channels());

            // 模型解析
            float[][] probMap = OnnxUtil.parseDetOutput(output);

            int h = probMap.length;
            int w = probMap[0].length;
            log.debug("概率图尺寸: {}x{}", w, h);

            // 保存结果
            context.setDetProbMap(probMap);
        } catch (OrtException e) {
            log.error("检测模型推理失败", e);
            throw e;
        }

        long elapsed = System.currentTimeMillis() - startTime;
        log.debug("检测模型解析完成, 耗时: {} ms", elapsed);
    }

    /**
     * 后处理：二值化、轮廓查找、文本框提取
     */
    private void postprocess(OCRContext context) {
        long startTime = System.currentTimeMillis();
        log.debug("开始检测后处理");

        // 获取概率图尺寸
        float[][] probMap = context.getDetProbMap();
        int h = probMap.length;
        int w = probMap[0].length;

        // 概率图转 Mat
        Mat prob = new Mat(h, w, CvType.CV_32FC1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                prob.put(y, x, probMap[y][x]);
            }
        }

        // 二值化，概率图 > 阈值 的区域为文本区域
        float detThresh = ocrConfig.getDetThresh();
        Mat bitmap = new Mat();
        Imgproc.threshold(prob, bitmap, detThresh, 255, Imgproc.THRESH_BINARY);
        log.debug("二值化完成, 阈值: {}", detThresh);

        // 转换为 8位 单通道
        bitmap.convertTo(bitmap, CvType.CV_8UC1);

        // 可选膨胀操作, 用于连接相邻的文本区域
        if (ocrConfig.isDilation()) {
            int kernelSize = modelConfig.getDilateKernelSize();
            Mat kernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT, new Size(kernelSize, kernelSize));
            Imgproc.dilate(bitmap, bitmap, kernel);
            OpenCVUtil.releaseMat(kernel);
            log.debug("膨胀操作完成, 核大小: {}", kernelSize);
        }

        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bitmap, contours, hierarchy,
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);

        int contourCount = contours.size();
        log.debug("轮廓检测完成, 原始轮廓数: {}", contourCount);

        // 限制候选框数量
        if (contours.size() > modelConfig.getMaxCandidates()) {
            contours = new ArrayList<>(contours.subList(0, modelConfig.getMaxCandidates()));
            log.debug("轮廓数量超过限制({}), 截取前{}个",
                    modelConfig.getMaxCandidates(), modelConfig.getMaxCandidates());
        }

        // 遍历每个轮廓，提取文本框
        List<TextBox> boxes = new ArrayList<>();
        int skippedByArea = 0;
        int skippedByScore = 0;
        int skippedByPerimeter = 0;
        int skippedBySize = 0;
        int skippedByExpand = 0;

        for (MatOfPoint contour : contours) {
            // 面积过滤
            double area = Imgproc.contourArea(contour);
            if (area <= 1.0) {
                skippedByArea++;
                OpenCVUtil.releaseMat(contour);
                continue;
            }

            // 计算轮廓区域内概率图的平均置信度并过滤
            double score = OpenCVUtil.getScore(contour, probMap);
            float boxThresh = ocrConfig.getDetBoxThresh();
            if (score < boxThresh) {
                skippedByScore++;
                OpenCVUtil.releaseMat(contour);
                continue;
            }

            // 获取轮廓点并计算周长
            Point[] contourPoints = contour.toArray();
            MatOfPoint2f contour2f = new MatOfPoint2f(contourPoints);
            double perimeter = Imgproc.arcLength(contour2f, true);
            if (perimeter < 1e-6) {
                skippedByPerimeter++;
                OpenCVUtil.releaseMat(contour2f);
                OpenCVUtil.releaseMat(contour);
                continue;
            }

            // unclip 扩张公式 距离 = 面积 * 扩张比率 / 周长
            double unclipRatio = ocrConfig.getDetUnclipRatio();
            double distance = area * unclipRatio / perimeter;
            List<Point> expanded = OpenCVUtil.unclipPolygon(contourPoints, distance);
            OpenCVUtil.releaseMat(contour2f);

            // 扩张后点数不足
            if (expanded.size() < 4) {
                skippedByExpand++;
                OpenCVUtil.releaseMat(contour);
                continue;
            }

            // 多边形近似（平滑轮廓）
            MatOfPoint2f expanded2f = new MatOfPoint2f(expanded.toArray(new Point[0]));
            double epsilon = modelConfig.getEpsilon() * Imgproc.arcLength(expanded2f, true);
            List<Point> approx = OpenCVUtil.approxPolyDP(expanded, epsilon, true);

            // 根据配置选择返回多边形还是最小外接矩形
            List<Point> contourPoint;
            if (ocrConfig.isDetUsePolygon()) {
                // 返回多边形（倾斜文本框）
                contourPoint = approx;
            } else {
                // 返回最小外接矩形（水平矩形）
                if (approx.size() < 4) {
                    skippedByExpand++;
                    OpenCVUtil.releaseMat(expanded2f);
                    OpenCVUtil.releaseMat(contour);
                    continue;
                }
                MatOfPoint2f approx2f = new MatOfPoint2f(approx.toArray(new Point[0]));
                RotatedRect rr = Imgproc.minAreaRect(approx2f);
                OpenCVUtil.releaseMat(approx2f);
                Point[] vertices = new Point[4];
                rr.points(vertices);
                contourPoint = OpenCVUtil.orderPoints(Arrays.asList(vertices));
            }

            // 最小尺寸过滤
            MatOfPoint2f box2f = new MatOfPoint2f(contourPoint.toArray(new Point[0]));
            RotatedRect sizeRect = Imgproc.minAreaRect(box2f);
            OpenCVUtil.releaseMat(box2f);
            OpenCVUtil.releaseMat(expanded2f);

            float minSize = ocrConfig.getDetMinSize();
            if (Math.min(sizeRect.size.width, sizeRect.size.height) < minSize) {
                skippedBySize++;
                OpenCVUtil.releaseMat(contour);
                continue;
            }

            // 坐标还原
            float scaleX = context.getDetPrepScaleX();
            float scaleY = context.getDetPrepScaleY();
            int rawW = context.getRawMat().width();
            int rawH = context.getRawMat().height();

            List<Point> restorePoints = OpenCVUtil.restorePoints(
                    contourPoint, scaleX, scaleY, rawW, rawH);

            // 裁剪
            Mat restoreMat = ocrConfig.isDetUsePolygon()
                    ? OpenCVUtil.polygonCrop(context.getRawMat(), restorePoints)
                    : OpenCVUtil.perspectiveTransformCrop(context.getRawMat(), restorePoints);

            TextBox box = TextBox.builder()
                    .contourPoint(contourPoint)
                    .contourMat(contour)
                    .restorePoints(restorePoints)
                    .restoreMat(restoreMat)
                    .build();
            boxes.add(box);
        }
        // 资源释放
        OpenCVUtil.releaseMat(hierarchy);
        OpenCVUtil.releaseMat(bitmap);
        OpenCVUtil.releaseMat(prob);
        // 设置结果
        context.setDetResultBoxes(boxes);

        long elapsed = System.currentTimeMillis() - startTime;

        // 输出统计信息
        log.info("检测后处理完成, 耗时: {} ms", elapsed);
        log.info("文本框统计 - 总轮廓: {}, 有效文本框: {}", contours.size(), boxes.size());
        log.info("过滤统计 - 面积不足: {}, 置信度不足: {}, 周长异常: {}, 扩张失败: {}, 尺寸不足: {}",
                skippedByArea, skippedByScore, skippedByPerimeter, skippedByExpand, skippedBySize);
    }

}
