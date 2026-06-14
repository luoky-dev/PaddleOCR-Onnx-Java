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

    /**
     * 图像检测 - 主流程
     */
    public void detect(OCRContext context) {
        log.debug("开始图像检测");
        long startTime = System.currentTimeMillis();
        try {
            // 预处理
            preprocess(context);
            // 模型推理
            parse(context);
            // 后处理
            postprocess(context);
        } catch (Exception e) {
            log.error("图像检测失败, 错误信息:",e);
            throw new RuntimeException("Runtime error, recognition failed");
        }
        log.debug("图像检测完成, 检测成功检测框数量: {}, 耗时: {} ms", context.getDetResultBoxes().size(), System.currentTimeMillis() - startTime);
    }

    /**
     * 图像检测 - 预处理
     * 将原图像转换为模型输入格式
     */
    private void preprocess(OCRContext context) throws OrtException {
        log.debug("图像检测 - 预处理阶段");
        long startTime = System.currentTimeMillis();
        // det模型输入形状和原图
        Mat rawMat = context.getRawMat();
        if (rawMat == null || rawMat.empty()) {
            log.error("图片不合法, 图像检测失败");
            throw new RuntimeException("The image is invalid, detection failed");
        }
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
        // 动态输入模型: 使用原始图像尺寸的最大边长
        int limitSize = modelInputShape[2] != -1 && modelInputShape[3] != -1 ?
                Math.max(Math.toIntExact(modelInputShape[2]), Math.toIntExact(modelInputShape[3])) :
                Math.max(rawMat.height(), rawMat.width());
        log.debug("长边限制大小: {}", limitSize);
        // 长边限制 + 对齐
        Size targetSize = OpenCVUtil.longSideLimitToStride(rawSize, limitSize, modelConfig.getStride());
        // 缩放图像 + 转换RGB通道
        Mat rgbMat = OpenCVUtil.resizeToRGB(rawMat, targetSize);
        log.debug("图像缩放完成: H:{} x W:{} -> H:{} x W:{}",
                rawMat.height(), rawMat.width(), rgbMat.height(), rgbMat.width());
        // 填充
        int modelInputH = Math.toIntExact(modelInputShape[3] == -1 ? (long) targetSize.height : modelInputShape[3]);
        int modelInputW = Math.toIntExact(modelInputShape[2] == -1 ? (long) targetSize.width : modelInputShape[2]);
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
        log.debug("预处理阶段完成, 耗时: {} ms", elapsed);
        context.setDetState(
                DetState.builder()
                        .chwData(chwData)
                        .rawMatSize(rawSize)
                        .resizeMatSize(targetSize)
                        .modelInputSize(modelInputSize).build()
        );
    }

    /**
     * 图像检测 - 模型推理
     * 进行模型推理
     */
    private void parse(OCRContext context) throws OrtException {
        log.debug("图像检测 - 模型推理阶段");
        long startTime = System.currentTimeMillis();
        DetState detState = context.getDetState();
        // 模型解析输入
        List<float[]> chwList = List.of(detState.getChwData());
        // 模型解析
        try (OnnxTensor input = OnnxUtil.createBatchInputTensor(chwList, modelManager.getEnv(), detState.getModelInputSize());
             Result output = modelManager.getDetSession().run(Collections.singletonMap("x", input))) {
            // 模型输出
            float[][] prob = OnnxUtil.parseOnnxValue2D(output);
            detState.setProb(prob);
            log.debug("模型推理完成, 特征图尺寸: Height:{} x Width:{}", prob.length, prob[0].length);
        } catch (OrtException e) {
            log.error("检测模型推理失败", e);
            throw e;
        }
        log.debug("模型推理阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }
    
    /**
     * 图像检测 - 后处理
     * 解码输出并进行检测框过滤和提取
     */
    private void postprocess(OCRContext context) {
        log.debug("图像检测 - 后处理检测框提取阶段");
        long startTime = System.currentTimeMillis();
        DetState detState = context.getDetState();
        // 概率图转 Mat
        float[][] probMap = detState.getProb();
        Mat probMat = OpenCVUtil.buildProbMat(probMap);
        // 查找轮廓
        List<MatOfPoint> contours = findContours(probMat);
        if (contours.isEmpty()) {
            log.error("轮廓检测完成, 未检测出轮廓框, 图像识别失败");
            throw new RuntimeException("No contour box detected, recognition failed");
        }
        // 限制候选框数量
        if (contours.size() > ocrConfig.getBoxLimit()) {
            contours = new ArrayList<>(contours.subList(0, ocrConfig.getBoxLimit()));
            log.debug("超过轮廓框数量限制, 保留前 {} 个", ocrConfig.getBoxLimit());
        }
        // 轮廓解析计算
        List<TextBox> textBoxes = parseContours(contours, probMat, detState);
        // 设值
        context.setDetResultBoxes(textBoxes);
        // 资源释放
        OpenCVUtil.releaseMat(probMat);
        log.debug("后处理检测框提取阶段完成, 耗时: {} ms", System.currentTimeMillis() - startTime);
    }

    /**
     * 轮廓框检测
     */
    private List<MatOfPoint> findContours(Mat probMat){
        log.debug("开始轮廓检测");
        // 1.二值化: 概率图 > 阈值 的区域为文本区域
        Mat bitmap = OpenCVUtil.threshold(probMat, ocrConfig.getBitThresh());
        log.debug("二值化完成, 阈值: {}", ocrConfig.getBitThresh());

        // 2.可选膨胀操作, 用于连接相邻的文本区域
        if (ocrConfig.isDilation()) {
            int kernelSize = modelConfig.getDilateKernelSize();
            Mat kernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT, new Size(kernelSize, kernelSize));
            Imgproc.dilate(bitmap, bitmap, kernel);
            OpenCVUtil.releaseMat(kernel);
            log.debug("膨胀操作完成, 核大小: {}", kernelSize);
        }

        // 3.查找轮廓
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
     * 轮廓框过滤解析
     */
    private List<TextBox> parseContours(List<MatOfPoint> contours, Mat probMat, DetState detState){
        log.debug("开始轮廓框解析过滤");
        // 设置检测结果
        List<TextBox> textBoxes = new ArrayList<>();
        List<ContourBox> contourBoxes = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) {
            log.trace("当前处理第 {} 个轮廓框: ", i);
            MatOfPoint contour = contours.get(i);
            ContourBox contourBox = new ContourBox();
            contourBoxes.add(contourBox);
            // 初始顶点坐标
            Point[] points = contour.toArray();
            contourBox.setPoints(points);
            // 1.噪声过滤
            // 顶点数量过滤
            log.trace("当前轮廓框顶点数量: {}", points.length);
            if (points.length < 4) {
                log.trace("顶点数量不足, 已过滤");
                contourBox.setNoiseFilter(true);
                continue;
            }
            double area = Imgproc.contourArea(contour);
            contourBox.setArea(area);
            log.trace("当前轮廓框面积: {}, 最小面积阈值: {}", area, ocrConfig.getBoxMinArea());
            // 面积过滤
            if (area <= ocrConfig.getBoxMinArea()) {
                log.trace("面积不足, 已过滤");
                contourBox.setNoiseFilter(true);
                continue;
            }
            // 边界框尺寸过滤
            Rect rect = Imgproc.boundingRect(contour);
            double maxSize = Math.max(rect.width, rect.height);
            contourBox.setBoundingRect(rect);
            log.trace("当前轮廓框外接矩形: H:{} x W:{}, 长边尺寸: {}, 最小尺寸阈值: {}",
                    rect.height, rect.width, maxSize, ocrConfig.getBoxMinSize());
            // 长边尺寸判断过滤
            if (maxSize < ocrConfig.getBoxMinSize()) {
                log.trace("长边尺寸不足, 已过滤");
                contourBox.setNoiseFilter(true);
                continue;
            }

            // 2.多边形近似
            Point[] approx = OpenCVUtil.approxPolyDP(contour, ocrConfig.getEpsilon(), true);
            contourBox.setApproxPoints(approx);
            log.trace("当前轮廓框多边近似完成, 腐蚀度: {}, 顶点数量: {} -> {}",
                    ocrConfig.getEpsilon(), contour.toArray().length, approx.length);
            // 多边近似失败过滤
            if (approx.length < 4) {
                log.trace("多边近似后顶点数量不足, 已过滤");
                contourBox.setApproxFilter(true);
                continue;
            }

            // 3.四边拟合
            Point[] quadPoints;
            if (approx.length > 4) {
                // 获取最小外接矩形顶点
                quadPoints = OpenCVUtil.minAreaRect(approx);
            } else {
                quadPoints = approx;
            }
            contourBox.setQuadPoints(quadPoints);
            log.trace("当前轮廓框四边拟合完成, 顶点坐标: {}", Arrays.asList(quadPoints));

            // 4.顶点排序
            quadPoints = OpenCVUtil.orderPoints(quadPoints);
            contourBox.setOrderPoints(quadPoints);
            log.trace("当前轮廓框顶点排序完成, 顶点坐标: {}", Arrays.asList(quadPoints));

            // 5.计算置信度
            double score = OpenCVUtil.getScore(contour, probMat);
            contourBox.setScore(score);
            log.trace("当前轮廓框平均置信度: {}, 最小置信度阈值: {}", score, ocrConfig.getBoxThresh());
            // 置信度过滤
            if (score < ocrConfig.getBoxThresh()) {
                log.trace("平均置信度不足, 已过滤");
                contourBox.setScoreFilter(true);
                continue;
            }

            // 6.扩张
            Point[] unclipPoints = OpenCVUtil.unclip(quadPoints, ocrConfig.getUnclipRatio());
            contourBox.setUnclipPoints(unclipPoints);
            log.trace("当前轮廓框扩张完成, 扩张比率: {}", ocrConfig.getUnclipRatio());
            log.trace("扩张前顶点: {}", Arrays.asList(quadPoints));
            log.trace("扩张后顶点: {}", Arrays.asList(unclipPoints));

            // 7.坐标还原
            Point[] restorePoints = OpenCVUtil.restorePoints(
                    unclipPoints, detState.getResizeMatSize(), detState.getRawMatSize());
            contourBox.setRestorePoints(restorePoints);
            log.trace("当前轮廓框坐标已还原到原图, 缩放图: H:{} x W:{}, 原图: H:{} x W:{}",
                    detState.getResizeMatSize().height, detState.getResizeMatSize().width,
                    detState.getRawMatSize().height, detState.getRawMatSize().width);
            log.trace("还原前顶点: {}", Arrays.asList(unclipPoints));
            log.trace("还原后顶点: {}", Arrays.asList(restorePoints));

            // 8.纵形轮廓框判断
            Size rectSize = OpenCVUtil.getRectSize(restorePoints);
            double aspectRatio = rectSize.width / Math.max(1, rectSize.height);
            contourBox.setAspectRatio(aspectRatio);
            log.trace("当前四边形轮廓框宽高比: {}, 最低宽高比阈值: {}", aspectRatio, ocrConfig.getBoxMinAspectRatio());

            // 9.设值
            TextBox box = new TextBox();
            box.setPoints(restorePoints);
            box.setAspectRatio(aspectRatio);
            box.setScore(score);
            if (aspectRatio < ocrConfig.getBoxMinAspectRatio()) {
                box.setAngle(90);
                box.setRotate(true);
            }
            textBoxes.add(box);
        }
        // 按模型输出的倒序排列
        Collections.reverse(textBoxes);
        // 设置序号
        for (int index = 1; index <= textBoxes.size(); index++) {
            textBoxes.get(index - 1).setIndex(index);
        }
        // 输出统计信息
        log.debug("结果统计 - 总轮廓框: {}, 有效检测框: {}", contours.size(), textBoxes.size());
        log.debug("过滤统计 - 噪声框过滤: {}, 置信度不足过滤: {}, 多边近似失败过滤: {}",
                contourBoxes.stream().filter(ContourBox::isNoiseFilter).count(),
                contourBoxes.stream().filter(ContourBox::isScoreFilter).count(),
                contourBoxes.stream().filter(ContourBox::isApproxFilter).count());
        // 资源释放
        contours.forEach(OpenCVUtil::releaseMat);
        detState.setContourBoxes(contourBoxes);
        return textBoxes;
    }
}