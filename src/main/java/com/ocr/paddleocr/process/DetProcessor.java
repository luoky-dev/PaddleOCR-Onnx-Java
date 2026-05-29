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

import java.util.*;
import java.util.stream.Collectors;

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
        log.info("开始图像检测");
        long startTime = System.currentTimeMillis();
        // 预处理
        preprocess(context);
        // 模型推理
        parse(context);
        // 后处理
        postprocess(context);
        log.info("图像检测完成, 检测成功检测框数量: {}, 耗时: {} ms", context.getDetResultBoxes().size(), System.currentTimeMillis() - startTime);
    }

    private void preprocess(OCRContext context) throws OrtException {
        log.info("图像检测 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // det模型输入形状和原图
        Mat rawMat = context.getRawMat();
        long[] modelInputShape = OnnxUtil.getModelInputShape(modelManager.getDetSession());
        // 判断模型输入
        if (modelInputShape[2] != modelInputShape[3]) {
            log.error("当前检测模型不支持, 图像检测失败");
            throw new RuntimeException("Detection model is not supported, detection failed");
        }
        log.debug("图像检测模型输入形状(-1代表动态输入): Batch: {} x Channel: {} x Height:{} x Width:{} ",
                modelInputShape[0], modelInputShape[1], modelInputShape[2], modelInputShape[3]);
        // 获取原始图像尺寸
        Size rawSize = new Size(rawMat.width(), rawMat.height());
        log.debug("原始图像尺寸: H:{} x W:{} ", rawMat.height(), rawMat.width());
        // 确定长边限制大小
        // 固定输入模型: 严格按模型声明尺寸送入
        // 动态输入模型: 使用配置的最大边长
        int limitSize = modelInputShape[2] != -1 && modelInputShape[3] != -1 ?
                Math.max(Math.toIntExact(modelInputShape[2]), Math.toIntExact(modelInputShape[3])) :
                ocrConfig.getDetModelMaxSide();
        log.debug("长边限制大小: {}", limitSize);
        // 长边限制 + 对齐
        Size targetSize = OpenCVUtil.longSideLimitToStride(rawSize, limitSize, modelConfig.getStride());
        // 缩放图像 + 转换RGB通道
        Mat rgbMat = OpenCVUtil.resizeToRGB(rawMat, targetSize);
        log.debug("图像缩放完成: H:{} x W:{} -> H:{} x W:{}",
                rawMat.height(), rawMat.width(), rgbMat.height(), rgbMat.width());
        // 填充
        int modelInputH = Math.toIntExact(modelInputShape[3] == -1 ? limitSize : modelInputShape[3]);
        int modelInputW = Math.toIntExact(modelInputShape[2] == -1 ? limitSize : modelInputShape[2]);
        Size modelInputSize = new Size(modelInputW, modelInputH);
        Mat paddedMat = OpenCVUtil.padding(rgbMat, modelInputSize);
        log.debug("图像填充完成: H:{} x W:{} -> H:{} x W:{}",
                rgbMat.height(), rgbMat.width(), paddedMat.height(), paddedMat.width());
        // 归一化 + 转换CHW格式数据
        float[] chwData = OpenCVUtil.normalizeToCHW(paddedMat, modelConfig.getScoreMean(), modelConfig.getScoreStd());
        log.debug("图像归一标准化完成, 均值: {}, 标准差: {}",
                Arrays.toString(modelConfig.getScoreMean()),
                Arrays.toString(modelConfig.getScoreStd()));
        // 资源释放
        OpenCVUtil.releaseMat(rgbMat);
        OpenCVUtil.releaseMat(paddedMat);
        // 返回结果
        long elapsed = System.currentTimeMillis() - startTime;
        log.info("预处理阶段完成, 耗时: {} ms", elapsed);
        context.setDetState(
                DetState.builder()
                        .chwData(chwData)
                        .rawMatSize(rawSize)
                        .resizeMatSize(targetSize)
                        .modelInputSize(modelInputSize).build()
        );
    }

    /**
     * 模型推理
     */
    private void parse(OCRContext context) throws OrtException {
        log.info("图像检测 - 模型推理阶段");
        long startTime = System.currentTimeMillis();
        DetState detState = context.getDetState();
        // 模型解析输入
        List<float[]> chwList = List.of(detState.getChwData());
        // 模型解析
        try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), detState.getModelInputSize());
             Result output = modelManager.getDetSession().run(Collections.singletonMap("x", input))) {
            // 模型输出
            float[][] prob = OnnxUtil.parseDetOutput(output);
            detState.setProb(prob);
            log.debug("模型推理完成, 特征图尺寸: Height:{} x Width:{}", prob.length, prob[0].length);
        } catch (OrtException e) {
            log.error("检测模型推理失败", e);
            throw e;
        }
        log.info("模型推理阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 后处理: 二值化、轮廓查找过滤、检测框提取
     */
    private void postprocess(OCRContext context) {
        log.info("图像检测 - 后处理检测框提取阶段");
        long startTime = System.currentTimeMillis();
        DetState detState = context.getDetState();
        // 概率图转 Mat
        float[][] probMap = detState.getProb();
        Mat probMat = OpenCVUtil.buildProbMat(probMap);
        // 查找轮廓
        List<MatOfPoint> contours = findContours(probMat);
        if (contours.isEmpty()) {
            log.error("轮廓检测完成, 未检测出轮廓, 图像识别失败");
            context.setDetResultBoxes(List.of());
        }
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
        long approxFilterCount = boxes.stream()
                .filter(ContourBox::isApproxFail).count();
        long sizeFilterCount = boxes.stream()
                .filter(ContourBox::isMinSizeFilter).count();
        long aspectRatioFilterCount = boxes.stream()
                .filter(ContourBox::isAspectRatioFilter).count();
        long expandFilterCount = boxes.stream()
                .filter(ContourBox::isUnclipFail).count();
        // 创建子目录
        String cropDir = ocrConfig.getDebugPath() + "/det_crops";
        OpenCVUtil.ensureDir(cropDir);
        // 按阅读顺序对检测框排序
        Map<Integer, Point[]> orderMap = OpenCVUtil.orderByRead(
                boxes.stream()
                        .map(ContourBox::getRestorePoints)
                        .filter(Objects::nonNull)
                        .collect(Collectors.toList())
        );
        // 设置检测结果
        List<TextBox> textBoxes = new ArrayList<>();
        orderMap.forEach( (index, points) -> textBoxes.add(
                TextBox.builder().
                        index(index).
                        points(points).
                        build()));
        orderMap.forEach( (index, points) -> {
            // 裁剪
            Mat cropMat = OpenCVUtil.perspectiveTransformCrop(context.getRawMat(), points);
            String file = String.format(Locale.ROOT, "%s/det_crop_%03d.jpg", cropDir, index);
            OpenCVUtil.saveImage(cropMat, file);
        });

        context.setDetResultBoxes(textBoxes);
        // 资源释放
        OpenCVUtil.releaseMat(probMat);
        // 输出统计信息
        log.debug("检测框统计 - 总轮廓框: {}, 有效检测框: {}", contours.size(), orderMap.size());
        log.debug("过滤统计 - 面积不足过滤: {}, 平均置信度不足过滤: {}, 多边近似失败: {}, 最大宽高比过滤: {}, 最小尺寸过滤: {}, 扩边失败: {}, ",
                areaFilterCount, scoreFilterCount, approxFilterCount, aspectRatioFilterCount, sizeFilterCount, expandFilterCount);
        log.info("后处理检测框提取阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 轮廓检测
     */
    private List<MatOfPoint> findContours(Mat probMat){
        log.debug("开始轮廓检测");
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

        log.debug("轮廓检测完成, 原始轮廓数: {}", contours.size());
        // 资源释放
        OpenCVUtil.releaseMat(hierarchy);
        OpenCVUtil.releaseMat(bitmap);
        return contours;
    }

    /**
     * 轮廓解析
     */
    private void parseContours(List<MatOfPoint> contours, Mat probMat, DetState detState){
        log.debug("开始轮廓解析过滤");
        List<ContourBox> contourBoxes = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            ContourBox contourBox = new ContourBox();
            log.info("当前处理第 {} 个轮廓框, 当前轮廓框顶点数量: {}", i, contour.toArray().length);
            // 计算面积
            double area = Imgproc.contourArea(contour);
            contourBox.setArea(area);
            log.info("当前轮廓框面积: {}", area);
            // 面积过滤
            if (area <= ocrConfig.getDetMinArea()) {
                log.info("最小面积阈值: {}, 当前轮廓框面积不足, 已过滤", ocrConfig.getDetMinArea());
                contourBox.setAreaFilter(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 计算平均置信度
            double score = OpenCVUtil.getScore(contour, probMat);
            contourBox.setScore(score);
            log.info("当前轮廓框平均置信度: {}", score);
            // 平均置信度过滤
            if (score < ocrConfig.getDetBoxThresh()) {
                log.info("最小平均置信度阈值: {}, 当前轮廓框平均置信度不足, 已过滤", ocrConfig.getDetBoxThresh());
                contourBox.setScoreFilter(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 多边形近似（平滑轮廓）
            Point[] approx = OpenCVUtil.approxPolyDP(contour, modelConfig.getEpsilon(), true);
            contourBox.setApproxPoints(approx);
            // 最终过滤后返回的四边形顶点
            Point[] quadPoints = new Point[4];
            log.info("当前轮廓框多边近似完成, 腐蚀度: {}, 顶点数量: {}", modelConfig.getEpsilon(), approx.length);
            if (approx.length < 4) {
                log.info("当前轮廓框多边近似完成后顶点数量不足, 已过滤");
                contourBox.setApproxFail(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 使用四边形拟合返回最小外接矩形顶点
            if (approx.length > 4) {
                // 获取最小外接矩形顶点
                MatOfPoint2f approx2f = new MatOfPoint2f(approx);
                RotatedRect rr = Imgproc.minAreaRect(approx2f);
                OpenCVUtil.releaseMat(approx2f);
                // 设置排序顶点
                rr.points(quadPoints);
                log.info("已使用四边形拟合, 返回当前轮廓框最小外接矩阵顶点");
            }
            // 顶点数量为4时将四边形框转换为矩形框
            if (approx.length == 4) {
                // 设置顶点排序
                quadPoints = approx;
            }
            // 获取过滤后的四边形最大尺寸
            Size rectSize = OpenCVUtil.getRectSize(quadPoints);
            // 计算宽高比
            double aspectRatio = rectSize.width / Math.max(1, rectSize.height);
            contourBox.setAspectRatio(aspectRatio);
            log.info("当前四边形轮廓框宽高比: {}", aspectRatio);
            // 最大宽高比过滤
            if (aspectRatio > ocrConfig.getDetMaxAspectRatio()) {
                log.info("最大宽高比阈值: {}, 当前四边形轮廓框宽高比过高, 已过滤", ocrConfig.getDetMinSize());
                contourBox.setAspectRatioFilter(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 计算最小尺寸
            double minSize = Math.min(rectSize.width, rectSize.height);
            contourBox.setMinSize(minSize);
            log.info("当前四边形轮廓框最小边尺寸: {}", minSize);
            // 最小尺寸过滤
            if (minSize < ocrConfig.getDetMinSize()) {
                log.info("最小尺寸阈值: {}, 当前四边形轮廓框最小边尺寸不足, 已过滤", ocrConfig.getDetMinSize());
                contourBox.setMinSizeFilter(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 轮廓框扩张
            // 扩张后的顶点
            Point[] unclipPoints = OpenCVUtil.unclipPolygon(quadPoints, ocrConfig.getDetUnclipRatio());
            Size rectSize1 = OpenCVUtil.getRectSize(unclipPoints);
            contourBox.setUnclipPoints(unclipPoints);
            log.info("当前轮廓框多边扩张完成, 扩张比率: {}, 扩张后顶点数量: {}",
                    ocrConfig.getDetUnclipRatio(), unclipPoints.length);
            if (unclipPoints.length < 4) {
                log.info("当前轮廓框扩张后顶点数量不足, 已过滤");
                contourBox.setUnclipFail(true);
                OpenCVUtil.releaseMat(contour);
                contourBoxes.add(contourBox);
                continue;
            }
            // 坐标还原
            Point[] restorePoints = OpenCVUtil.restorePoints(
                    unclipPoints, detState.getResizeMatSize(), detState.getRawMatSize());
            log.trace("轮廓框坐标已还原到原图");
            log.trace("还原前顶点: {}", Arrays.asList(unclipPoints));
            log.trace("还原后顶点: {}", Arrays.asList(restorePoints));
            contourBox.setRestorePoints(restorePoints);
            // 资源释放
            OpenCVUtil.releaseMat(contour);
            contourBoxes.add(contourBox);
        }
        detState.setContourBoxes(contourBoxes);
    }

//    /**
//     * 轮廓解析
//     */
//    private void parseContours1(List<MatOfPoint> contours, Mat probMat, DetState detState){
//        log.debug("开始轮廓解析过滤");
//        List<ContourBox> contourBoxes = new ArrayList<>();
//        for (int i = 0; i < contours.size(); i++) {
//            MatOfPoint contour = contours.get(i);
//            ContourBox contourBox = new ContourBox();
//            log.trace("当前处理第 {} 个轮廓框, 当前轮廓框顶点数量: {}", i, contour.toArray().length);
//            // 1.多边形近似
//            Point[] approx = OpenCVUtil.approxPolyDP(contour, modelConfig.getEpsilon(), true);
//            contourBox.setApproxPoints(approx);
//            // 最终过滤后返回的四边形顶点
//            Point[] quadPoints;
//            log.trace("当前轮廓框多边近似完成, 腐蚀度: {}, 顶点数量: {} -> {}",
//                    modelConfig.getEpsilon(), contour.toArray().length, approx.length);
//            // 顶点数量不足过滤
//            if (approx.length < 4) {
//                log.trace("当前轮廓框多边近似完成后顶点数量不足, 已过滤");
//                contourBox.setApproxFail(true);
//                OpenCVUtil.releaseMat(contour);
//                contourBoxes.add(contourBox);
//                continue;
//            }
//            // 2.四边拟合
//            else if (approx.length > 4) {
//                // 获取最小外接矩形顶点
//                quadPoints = OpenCVUtil.minAreaRect(approx);
//                log.trace("已使用四边形拟合, 返回当前轮廓框最小外接矩阵顶点");
//            } else {
//                quadPoints = approx;
//            }
//            // 3.顶点排序
//            quadPoints = OpenCVUtil.orderPoints(quadPoints);
//            // 扩张前
//            Size rectSize1 = OpenCVUtil.getRectSize(quadPoints);
//
//            // 4.计算置信度
//            double score = OpenCVUtil.getScore(contour, probMat);
//            contourBox.setScore(score);
//            log.trace("当前轮廓框平均置信度: {}", score);
//            // 置信度过滤
//            if (score < ocrConfig.getDetBoxThresh()) {
//                log.trace("最小平均置信度阈值: {}, 当前轮廓框平均置信度不足, 已过滤", ocrConfig.getDetBoxThresh());
//                contourBox.setScoreFilter(true);
//                OpenCVUtil.releaseMat(contour);
//                contourBoxes.add(contourBox);
//                continue;
//            }
//            // 5.扩张
//            Point[] unclipPoints = OpenCVUtil.unclipByDistance(quadPoints, ocrConfig.getDetUnclipRatio());
//            // 扩张后
//            Size rectSize2 = OpenCVUtil.getRectSize(unclipPoints);
//
//            log.trace("当前轮廓框扩张完成, 扩张比率: {}, 扩张后顶点数量: {}",
//                    ocrConfig.getDetUnclipRatio(), unclipPoints.length);
//            if (unclipPoints.length < 4) {
//                log.trace("当前轮廓框扩张后顶点数量不足, 已过滤");
//                contourBox.setUnclipFail(true);
//                OpenCVUtil.releaseMat(contour);
//                contourBoxes.add(contourBox);
//                continue;
//            }
//            // 6.坐标还原
//            Point[] restorePoints = OpenCVUtil.restorePoints(
//                    unclipPoints, detState.getResizeMatSize(), detState.getRawMatSize());
//            log.trace("轮廓框坐标已还原到原图");
//            log.trace("还原前顶点: {}", Arrays.asList(unclipPoints));
//            log.trace("还原后顶点: {}", Arrays.asList(restorePoints));
//
//            // 7.计算面积
//            double area = OpenCVUtil.getArea(restorePoints);
//            contourBox.setArea(area);
//            log.trace("当前轮廓框面积: {}", area);
//            // 面积过滤
//            if (area <= ocrConfig.getDetMinArea()) {
//                log.trace("最小面积阈值: {}, 当前轮廓框面积不足, 已过滤", ocrConfig.getDetMinArea());
//                contourBox.setAreaFilter(true);
//                OpenCVUtil.releaseMat(contour);
//                contourBoxes.add(contourBox);
//                continue;
//            }
//            // 获取过滤后的四边形最大尺寸
//            Size rectSize = OpenCVUtil.getRectSize(restorePoints);
//            // 8.计算宽高比
//            double aspectRatio = rectSize.width / Math.max(1, rectSize.height);
//            contourBox.setAspectRatio(aspectRatio);
//            log.trace("当前四边形轮廓框宽高比: {}", aspectRatio);
//            // 最大宽高比过滤
//            if (aspectRatio > ocrConfig.getDetMaxAspectRatio()) {
//                log.trace("最大宽高比阈值: {}, 当前四边形轮廓框宽高比过高, 已过滤", ocrConfig.getDetMinSize());
//                contourBox.setAspectRatioFilter(true);
//                OpenCVUtil.releaseMat(contour);
//                contourBoxes.add(contourBox);
//                continue;
//            }
//            // 最小宽高比顶点重新排序
//            if (aspectRatio < ocrConfig.getDetMinAspectRatio()) {
//                restorePoints = OpenCVUtil.rotateOrderPoints(restorePoints);
//                log.trace("当前四边形轮廓框宽高比过低, 已进行顶点重新排序交换宽高");
//            }
//            // 9.计算最小尺寸
//            double minSize = Math.min(rectSize.width, rectSize.height);
//            contourBox.setMinSize(minSize);
//            log.trace("当前四边形轮廓框最小边尺寸: {}", minSize);
//            // 最小尺寸过滤
//            if (minSize < ocrConfig.getDetMinSize()) {
//                log.trace("最小尺寸阈值: {}, 当前四边形轮廓框最小边尺寸不足, 已过滤", ocrConfig.getDetMinSize());
//                contourBox.setMinSizeFilter(true);
//                OpenCVUtil.releaseMat(contour);
//                contourBoxes.add(contourBox);
//                continue;
//            }
//            // 设值和资源释放
//            contourBox.setRestorePoints(restorePoints);
//            OpenCVUtil.releaseMat(contour);
//            contourBoxes.add(contourBox);
//        }
//        detState.setContourBoxes(contourBoxes);
//    }


}
