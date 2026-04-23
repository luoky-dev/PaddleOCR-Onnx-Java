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
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DetProcessor {

    private final ModelManager modelManager;
    private final OCRConfig ocrConfig;
    private final ModelConfig modelConfig;

    public DetProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.ocrConfig = modelManager.getOcrConfig();
        this.modelConfig = modelManager.getModelConfig();
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
        // 获取原始图像尺寸
        Mat raw = context.getRawMat();
        int srcW = raw.cols();
        int srcH = raw.rows();
        // 计算缩放比例
        int maxSide = Math.max(srcW, srcH);
        float scale = maxSide > modelConfig.getDetMaxSideLen() ? (float) modelConfig.getDetMaxSideLen() / maxSide : 1.0f;
        // 限制最大边长，不填充，如果最大边长超过限制(如960)，按比例缩小；否则保持原尺寸
        // 计算缩放后尺寸
        int dstW = Math.max((int) (srcW * scale), modelConfig.getResizeAlign());
        int dstH = Math.max((int) (srcH * scale), modelConfig.getResizeAlign());
        // 对齐到32的倍数，DB模型的下采样倍数是32，输入尺寸必须能被32整除
        dstW = Math.max((dstW / modelConfig.getResizeAlign()) * modelConfig.getResizeAlign(), modelConfig.getResizeAlign());
        dstH = Math.max((dstH / modelConfig.getResizeAlign()) * modelConfig.getResizeAlign(), modelConfig.getResizeAlign());
        // 缩放图像，使用双线性插值，保持图像内容不变形
        Mat resized = new Mat();
        Imgproc.resize(raw, resized, new Size(dstW, dstH));
        // 官方检测模型输入要求 RGB 顺序
        Mat rgb = new Mat();
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        resized.release();
        // 归一化 + 标准化
        Mat normalized = OpenCVUtil.normalize(rgb,modelConfig.getMean(),modelConfig.getStd());
        rgb.release();
        // 保存结果
        context.setDetPrepMat(normalized);
        context.setDetPrepScale(scale);
    }

    private void parse(OCRContext context) throws OrtException {
        // 模型输出
        OnnxTensor input = OnnxUtil.createInputTensor(context.getDetPrepMat(), modelManager.getEnv());
        Result output = modelManager.getDetSession().run(Collections.singletonMap("x", input));
        // 模型解析
        float[][] probMap = OnnxUtil.parseDetOutput(output);
        // 保存结果
        context.setDetProbMap(probMap);
        output.close();
        input.close();
    }

    private void postprocess(OCRContext context) {
        // 获取概率图尺寸
        int h = context.getDetProbMap().length;
        int w = context.getDetProbMap()[0].length;

        // 概率图转 Mat
        Mat prob = new Mat(h, w, CvType.CV_32FC1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                prob.put(y, x, context.getDetProbMap()[y][x]);
            }
        }

        // 二值化，概率图 > 阈值 的区域为文本区域
        Mat bitmap = new Mat();
        Imgproc.threshold(prob, bitmap, ocrConfig.getDetThresh(), 255, Imgproc.THRESH_BINARY);
        // 转换为 8位 单通道
        bitmap.convertTo(bitmap, CvType.CV_8UC1);

        // 可选膨胀操作, 用于连接相邻的文本区域
        if (ocrConfig.isDilation()) {
            Mat kernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT, new Size(modelConfig.getDilateKernelSize(), modelConfig.getDilateKernelSize()));
            Imgproc.dilate(bitmap, bitmap, kernel);
            kernel.release();
        }

        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bitmap, contours, hierarchy,
                Imgproc.RETR_LIST,      // 检测所有轮廓，不建立层级关系
                Imgproc.CHAIN_APPROX_SIMPLE);  // 压缩水平/垂直/对角线段
        hierarchy.release();
        bitmap.release();
        prob.release();

        // 按面积降序排序
        contours.sort((a, b) -> Double.compare(Imgproc.contourArea(b), Imgproc.contourArea(a)));

        // 限制候选框数量
        if (contours.size() > modelConfig.getMaxCandidates()) {
            contours = new ArrayList<>(contours.subList(0, modelConfig.getMaxCandidates()));
        }

        // 遍历每个轮廓，提取文本框
        List<TextBox> boxes = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            // 面积过滤, 面积太小，跳过
            double area = Imgproc.contourArea(contour);
            if (area <= 1.0) continue;

            // 计算轮廓区域内概率图的平均置信度并过滤
            double score = DBPostProcess.getScore(contour, context.getDetProbMap());
            if (score < ocrConfig.getBoxThresh()) continue;

            // 获取轮廓点并计算周长
            Point[] contourPoints = contour.toArray();
            MatOfPoint2f contour2f = new MatOfPoint2f(contourPoints);
            double perimeter = Imgproc.arcLength(contour2f, true);
            if (perimeter < 1e-6) {
                contour2f.release();
                continue;
            }

            // unclip 扩张公式 距离 = 面积 * 扩张比率 / 周长
            double distance = area * ocrConfig.getDetUnclipRatio() / perimeter;
            List<Point> expanded = DBPostProcess.unclipPolygon(contourPoints, distance);
            contour2f.release();
            // 扩张后点数不足，跳过
            if (expanded.size() < 4) continue;

            // 多边形近似（平滑轮廓）
            MatOfPoint2f expanded2f = new MatOfPoint2f(expanded.toArray(new Point[0]));
            double epsilon = modelConfig.getEpsilon() * Imgproc.arcLength(expanded2f, true);
            List<Point> approx = DBPostProcess.approxPolyDP(expanded, epsilon, true);

            // 根据配置选择返回多边形还是最小外接矩形
            List<Point> contourPoint;
            MatOfPoint2f box2f;
            if (ocrConfig.isDetUsePolygon()) {
                // 返回多边形（倾斜文本框）
                contourPoint = approx;
            } else {
                // 返回最小外接矩形（水平矩形）
                if (approx.size() < 4) {
                    expanded2f.release();
                    continue;
                }
                MatOfPoint2f approx2f = new MatOfPoint2f(approx.toArray(new Point[0]));
                RotatedRect rr = Imgproc.minAreaRect(approx2f);
                approx2f.release();
                Point[] vertices = new Point[4];
                rr.points(vertices);
                contourPoint = OpenCVUtil.orderPoints(Arrays.asList(vertices));
            }
            box2f = new MatOfPoint2f(contourPoint.toArray(new Point[0]));

            // 最小尺寸过滤
            RotatedRect sizeRect = Imgproc.minAreaRect(box2f);
            box2f.release();
            expanded2f.release();
            if (Math.min(sizeRect.size.width, sizeRect.size.height) < ocrConfig.getDetMinSize()) continue;

            // 坐标还原
            List<Point> restorePoints = OpenCVUtil.restorePoints(
                    contourPoint,
                    context.getDetPrepScale(),
                    context.getRawMat().width(),
                    context.getRawMat().height());
            // 透视变换裁剪
            Mat restoreMat = OpenCVUtil.perspectiveTransformCrop(context.getRawMat(),restorePoints);

            TextBox box = TextBox.builder()
                    .contourPoint(contourPoint)
                    .contourMat(contour)
                    .restorePoints(restorePoints)
                    .restoreMat(restoreMat)
                    .build();
            boxes.add(box);
        }
        context.setDetResultBoxes(boxes);
    }
}
