package com.ocr.paddleocr.process;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.ContourBox;
import com.ocr.paddleocr.domain.DetState;
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
    }

    public void detect(OCRContext context) throws OrtException {
        log.info("图像检测");
        long startTime = System.currentTimeMillis();
        // 预处理
        DetState detState = preprocess(context.getRawMat());
        // 模型预检测
        parse(detState);
        // 后处理
        List<TextBox> detResultBoxes = postprocess(detState);
        // 设置结果
        long elapsed = System.currentTimeMillis() - startTime;
        context.setDetResultBoxes(detResultBoxes);
        context.setDetProcessTime(elapsed);
        log.info("图像检测完成, 检测框数量: {}, 检测处理时间: {} ms", detResultBoxes.size(), elapsed);
    }

    private DetState preprocess(Mat rawMat) throws OrtException {
        log.info("图像检测 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // det模型输入形状
        long[] modelInputShape = OnnxUtil.getModelInputShape(modelManager.getDetSession());
        log.debug("检测模型输入形状(-1代表动态输入): Batch: {} x Channel: {} x Height:{} x Width:{} ",
                modelInputShape[0], modelInputShape[1], modelInputShape[2], modelInputShape[3]);
        // 获取原始图像尺寸
        Size rawSize = new Size(rawMat.width(), rawMat.height());
        log.debug("原始图像尺寸: H:{} x W:{} ", rawMat.height(), rawMat.width());
        // 确定长边限制大小
        // 固定输入模型: 严格按模型声明尺寸送入
        // 动态输入模型: 使用配置的最大边长
        int limitSize = modelInputShape[2] != -1 && modelInputShape[3] != -1 ?
                Math.max(Math.toIntExact(modelInputShape[2]), Math.toIntExact(modelInputShape[3])) :
                modelConfig.getDetMaxSide();
        log.debug("长边限制大小: {}", limitSize);
        // 长边限制 + 对齐
        Size targetSize = OpenCVUtil.longSideLimitToStride(rawSize, limitSize, modelConfig.getDetStride());
        // 缩放图像 + 转换RGB通道
        Mat rgbMat = OpenCVUtil.resizeToRGB(rawMat, targetSize);
        log.debug("图像缩放完成: H:{} x W:{} -> H:{} x W:{}",
                rawMat.height(), rawMat.width(), rgbMat.height(), rgbMat.width());
        // 如果是固定输入模型需要padding
        Mat paddedMat;
        if (modelInputShape[2] != -1 && modelInputShape[3] != -1) {
            paddedMat = OpenCVUtil.padding(rawMat, new Size(modelInputShape[3], modelInputShape[2]));
            log.debug("图像填充完成: H:{} x W:{} -> H:{} x W:{}",
                    rgbMat.height(), rgbMat.width(), paddedMat.height(), paddedMat.width());
        } else {
            paddedMat = rgbMat;
        }
        // 归一化 + 转换CHW格式数据
        float[] chwData = OpenCVUtil.normalizeToCHW(paddedMat, modelConfig.getScoreMean(), modelConfig.getScoreStd());
        Size modelInputSize = new Size(paddedMat.width(), paddedMat.height());
        log.debug("图像归一标准化完成, 均值: {}, 标准差: {}",
                Arrays.toString(modelConfig.getScoreMean()),
                Arrays.toString(modelConfig.getScoreStd()));
        // 资源释放
        OpenCVUtil.releaseMat(rgbMat);
        OpenCVUtil.releaseMat(paddedMat);
        // 返回结果
        long elapsed = System.currentTimeMillis() - startTime;
        log.info("预处理阶段完成, 耗时: {} ms", elapsed);
        return DetState.builder()
                .chwData(chwData)
                .rawMatSize(rawSize)
                .resizeMatSize(targetSize)
                .modelInputSize(modelInputSize).build();
    }

    /**
     * 模型推理
     */
    private void parse(DetState detState) throws OrtException {
        log.info("图像检测 - 模型推理阶段");
        long startTime = System.currentTimeMillis();
        // 模型解析输入
        List<float[]> chwList = List.of(detState.getChwData());
        // 模型解析
        try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), detState.getModelInputSize());
             Result output = modelManager.getDetSession().run(Collections.singletonMap("x", input))) {
            // 模型输出
            float[][] probMap = OnnxUtil.parseDetOutput(output);
            detState.setProbMap(probMap);
            log.debug("模型推理完成, 特征图尺寸: Height:{} x Width:{}", probMap.length, probMap[0].length);
        } catch (OrtException e) {
            log.error("检测模型推理失败", e);
            throw e;
        }
        log.info("模型推理阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 后处理: 二值化、轮廓查找过滤、检测框提取
     */
    private List<TextBox> postprocess(DetState detState) {
        log.info("图像检测 - 后处理检测框提取阶段");
        long startTime = System.currentTimeMillis();
        // 概率图转 Mat
        float[][] probMap = detState.getProbMap();
        Mat probMat = OpenCVUtil.buildProbMat(probMap);
        // 查找轮廓
        List<MatOfPoint> contours = findContours(probMat);
        if (contours.isEmpty()) {
            log.error("轮廓检测完成, 未检测出轮廓, 图像识别失败");
            return Collections.emptyList();
        }
        log.debug("轮廓检测完成, 原始轮廓数: {}", contours.size());
        // 限制候选框数量
        if (contours.size() > modelConfig.getMaxCandidates()) {
            contours = new ArrayList<>(contours.subList(0, modelConfig.getMaxCandidates()));
            log.debug("轮廓数量超过限制({}), 截取前{}个",
                    modelConfig.getMaxCandidates(), modelConfig.getMaxCandidates());
        }
        // 轮廓解析计算
        parseContours(contours, probMat, detState);
        // 结果过滤
        List<ContourBox> boxes = detState.getContourBoxes();
        long areaFilterCount = boxes.stream()
                .filter(ContourBox::isAreaFilter).count();
        long scoreFilterCount = boxes.stream()
                .filter(ContourBox::isScoreFilter).count();
        long perimeterFilterCount = boxes.stream()
                .filter(ContourBox::isPerimeterFilter).count();
        long sizeFilterCount = boxes.stream()
                .filter(ContourBox::isMinSizeFilter).count();
        long expandFilterCount = boxes.stream()
                .filter(ContourBox::isUnclipFail).count();
        long approxFilterCount = boxes.stream()
                .filter(ContourBox::isApproxFail).count();
        // 设值过滤后还原的检测框
        int index = 0;
        List<TextBox> textBoxes = new ArrayList<>();
        for (ContourBox contourBox : boxes) {
            TextBox box = new TextBox();
            List<Point> points = contourBox.getRestorePoints();
            if (points != null) {
                box.setIndex(index);
                box.setPoints(points);
                box.setAspectRatio(contourBox.getAspectRatio());
                textBoxes.add(box);
                index++;
            }
        }
        // 资源释放
        OpenCVUtil.releaseMat(probMat);
        // 输出统计信息
        log.info("检测框统计 - 总轮廓框: {}, 有效检测框: {}", contours.size(), index);
        log.info("过滤统计 - 面积不足过滤: {}, 置信度不足过滤: {}, 周长异常过滤: {}, 最小尺寸过滤: {}, 扩边失败: {}, 多边近似失败: {}",
                areaFilterCount, scoreFilterCount, perimeterFilterCount, sizeFilterCount, expandFilterCount, approxFilterCount);
        log.info("后处理检测框提取阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
        return textBoxes;
    }

    /**
     * 轮廓检测
     */
    private List<MatOfPoint> findContours(Mat probMat){
        // 二值化，概率图 > 阈值 的区域为文本区域
        float detThresh = ocrConfig.getDetThresh();
        Mat bitmap = new Mat();
        Imgproc.threshold(probMat, bitmap, detThresh, 255, Imgproc.THRESH_BINARY);
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

        // 资源释放
        OpenCVUtil.releaseMat(hierarchy);
        OpenCVUtil.releaseMat(bitmap);
        return contours;
    }

    /**
     * 轮廓解析
     */
    private void parseContours(List<MatOfPoint> contours, Mat probMat, DetState detState){
        List<ContourBox> contourBoxes = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            ContourBox contourBox = new ContourBox();
            contourBox.setIndex(i);
            contourBox.setPoints(contour.toList());
            log.trace("当前处理第 {} 个轮廓框, 当前轮廓框顶点数量: {}", i, contourBox.getPoints().size());
            // 计算宽高比
            Rect rect = Imgproc.boundingRect(contour);
            double aspectRatio = (double) rect.width / Math.max(1, rect.height);
            contourBox.setAspectRatio(aspectRatio);
            log.trace("当前轮廓框宽高比: {}", aspectRatio);
            // 计算面积
            double area = Imgproc.contourArea(contour);
            contourBox.setArea(area);
            log.trace("当前轮廓框面积: {}", area);
            // 面积过滤
            if (area <= ocrConfig.getDetMinArea()) {
                log.trace("最小面积阈值: {}, 当前轮廓框面积不足, 已过滤", ocrConfig.getDetMinArea());
                contourBox.setAreaFilter(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 计算平均置信度
            double score = OpenCVUtil.getScore(contour, probMat);
            contourBox.setScore(score);
            log.trace("当前轮廓框平均置信度: {}", score);
            // 平均置信度过滤
            if (score < ocrConfig.getDetBoxThresh()) {
                log.trace("最小平均置信度阈值: {}, 当前轮廓框平均置信度不足, 已过滤", ocrConfig.getDetBoxThresh());
                contourBox.setScoreFilter(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 计算周长
            Point[] contourPoints = contour.toArray();
            MatOfPoint2f contour2f = new MatOfPoint2f(contourPoints);
            double perimeter = Imgproc.arcLength(contour2f, true);
            contourBox.setPerimeter(perimeter);
            log.trace("当前轮廓框周长: {}", perimeter);
            // 周长过滤
            if (perimeter < 1e-6) {
                log.trace("当前轮廓框周长异常, 鉴定为噪声点, 已过滤");
                contourBox.setScoreFilter(true);
                OpenCVUtil.releaseMat(contour);
                OpenCVUtil.releaseMat(contour2f);
                contourBoxes.add(contourBox);
                continue;
            }
            // 计算扩张点
            // unclip 扩张公式 距离 = 面积 * 扩张比率 / 周长
            double unclipRatio = ocrConfig.getDetUnclipRatio();
            double distance = area * unclipRatio / perimeter;
            List<Point> unclipPoints = OpenCVUtil.unclipPolygon(contourPoints, distance);
            contourBox.setUnclipPoints(unclipPoints);
            log.trace("当前轮廓框已扩张, 扩张比率: {}, 扩张后多边顶点数量: {}", unclipRatio, unclipPoints.size());
            if (unclipPoints.size() < 4) {
                log.trace("当前轮廓框扩张后多边顶点数量不足, 已过滤");
                contourBox.setUnclipFail(true);
                OpenCVUtil.releaseMat(contour);
                OpenCVUtil.releaseMat(contour2f);
                contourBoxes.add(contourBox);
                continue;
            }
            // 多边形近似（平滑轮廓）
            MatOfPoint2f unclip2f = new MatOfPoint2f(unclipPoints.toArray(new Point[0]));
            double epsilon = modelConfig.getEpsilon() * Imgproc.arcLength(unclip2f, true);
            List<Point> approx = OpenCVUtil.approxPolyDP(unclipPoints, epsilon, true);
            log.trace("当前轮廓框多边近似完成, 腐蚀度: {}, 顶点数量: {}", epsilon, approx.size());
            if (approx.size() < 4) {
                log.trace("当前轮廓框多边近似完成后顶点数量不足, 已过滤");
                contourBox.setApproxFail(true);
                OpenCVUtil.releaseMat(contour);
                OpenCVUtil.releaseMat(contour2f);
                OpenCVUtil.releaseMat(unclip2f);
                contourBoxes.add(contourBox);
                continue;
            }
            // 使用四边形拟合返回最小外接矩形顶点
            if (approx.size() > 4) {
                MatOfPoint2f approx2f = new MatOfPoint2f(approx.toArray(new Point[0]));
                RotatedRect rr = Imgproc.minAreaRect(approx2f);
                OpenCVUtil.releaseMat(approx2f);
                Point[] vertices = new Point[4];
                rr.points(vertices);
                approx = OpenCVUtil.orderPoints(Arrays.asList(vertices));
                log.trace("已使用四边形拟合, 返回当前轮廓框最小外接矩阵顶点");
            }
            // 计算最小尺寸
            MatOfPoint2f box2f = new MatOfPoint2f(approx.toArray(new Point[0]));
            RotatedRect sizeRect = Imgproc.minAreaRect(box2f);
            double minSize = Math.min(sizeRect.size.width, sizeRect.size.height);
            contourBox.setMinSize(minSize);
            log.trace("当前轮廓框最小边尺寸: {}", minSize);
            if (minSize < ocrConfig.getDetMinSize()) {
                log.trace("最小尺寸阈值: {}, 当前轮廓框最小边尺寸不足, 已过滤", ocrConfig.getDetMinSize());
                contourBox.setMinSizeFilter(true);
                OpenCVUtil.releaseMat(contour);
                OpenCVUtil.releaseMat(contour2f);
                OpenCVUtil.releaseMat(unclip2f);
                OpenCVUtil.releaseMat(box2f);
                contourBoxes.add(contourBox);
                continue;
            }
            // 坐标还原
            List<Point> restorePoints = OpenCVUtil.restorePoints(
                    approx, detState.getResizeMatSize(), detState.getRawMatSize());
            log.trace("轮廓框坐标已还原到原图");
            log.trace("还原前顶点: {}", approx);
            log.trace("还原后顶点: {}", restorePoints);
            contourBox.setRestorePoints(restorePoints);
            // 资源释放
            OpenCVUtil.releaseMat(contour);
            OpenCVUtil.releaseMat(contour2f);
            OpenCVUtil.releaseMat(unclip2f);
            OpenCVUtil.releaseMat(box2f);
            contourBoxes.add(contourBox);
        }
        detState.setContourBoxes(contourBoxes);
    }
}
