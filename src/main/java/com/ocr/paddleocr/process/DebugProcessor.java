package com.ocr.paddleocr.process;

import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.*;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
public class DebugProcessor {

    public static void printDebugImages(OCRContext context, OCRConfig config) {
        // debug目录
        String debugPath = config.getDebugPath() + "/" + context.getImageName().substring(0,context.getImageName().lastIndexOf(".")) + "_Debug";
        ensureDir(debugPath);

        safeRun("originalImage", () -> printOriginalImage(context, debugPath));
        safeRun("preDetImage", () -> printPreDetImage(context, debugPath));
        safeRun("heatmapImage", () -> printHeatMapImage(context, debugPath));
        safeRun("bitmapImage", () -> printBitMapImage(context, config, debugPath));
        safeRun("contourImage", () -> printContourImage(context, debugPath));
        safeRun("detBoxImage", () -> printDetBoxImage(context, debugPath));
        safeRun("cropFillImage", () -> printCropFillImage(context, debugPath));
        if (config.isUseCls() && config.isUseDebug()) {
            safeRun("clsCropImage", () -> printClsCropImage(context, debugPath));
            safeRun("clsResultImage", () -> printClsResultImage(context, debugPath));
        }
        safeRun("rotateImage", () -> printRotateImage(context, debugPath));
        safeRun("recCropImage", () -> printRecCropImage(context, debugPath));
        safeRun("recConfidenceImage", () -> printRecConfidenceImage(context, debugPath));
        safeRun("recResultImage", () -> printRecResultImage(context, debugPath));
    }

    public static void printOriginalImage(OCRContext context, String debugPath) {
        // 获取原图
        Mat rawMat = context.getRawMat();
        // 保存
        OpenCVUtil.saveImage(rawMat, debugPath + "/originalImage.jpg");
        log.debug("已保存原图像, 文件名: originalImage.jpg, 文件路径: {}", debugPath);
    }


    public static void printPreDetImage(OCRContext context, String debugPath) {
        // 获取检测预处理图像
        Mat preDetMat = getPreDetImage(context);
        // 保存
        OpenCVUtil.saveImage(preDetMat, debugPath + "/preDetImage.jpg");
        // 资源释放
        OpenCVUtil.releaseMat(preDetMat);
        log.debug("已保存检测阶段预处理后的图像, 文件名: preDetImage.jpg, 文件路径: {}", debugPath);
    }

    /**
     * 检测模型输出的概率热力图
     * 用途: 查看模型对文本区域的预测置信度分布
     * - 红色/黄色: 高概率 (文本区域)
     * - 蓝色/黑色: 低概率 (背景)
     */
    public static void printHeatMapImage(OCRContext context, String debugPath) {
        // 概率图转 Mat
        float[][] probMap = context.getDetState().getProb();
        Mat probMat = OpenCVUtil.buildProbMat(probMap);
        // 标准化
        Mat prob8 = new Mat();
        Core.normalize(probMat, prob8, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
        Mat heatmap = new Mat();
        Imgproc.applyColorMap(prob8, heatmap, Imgproc.COLORMAP_JET);
        // 保存
        OpenCVUtil.saveImage(heatmap, debugPath + "/heatmapImage.jpg");
        log.debug("已保存检测模型输出的概率热力图, 文件名: heatmapImage.jpg, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(prob8);
        OpenCVUtil.releaseMat(probMat);
        OpenCVUtil.releaseMat(heatmap);
    }

    /**
     * 概率图二值化后的图像
     * 用途: 查看经过阈值过滤后的文本区域
     * - 白色: 超过阈值的区域 (可能是文本)
     * - 黑色: 低于阈值的区域 (背景)
     */
    public static void printBitMapImage(OCRContext context, OCRConfig config, String debugPath) {
        // 获取阈值
        float threshold = config.getBitThresh();
        // 概率图转 Mat
        float[][] probMap = context.getDetState().getProb();
        Mat probMat = OpenCVUtil.buildProbMat(probMap);
        // 创建二值图
        Mat binary = OpenCVUtil.threshold(probMat, threshold);
        // 保存
        OpenCVUtil.saveImage(binary, debugPath + "/bitmapImage.jpg");
        // 资源释放
        OpenCVUtil.releaseMat(probMat);
        OpenCVUtil.releaseMat(binary);
        log.debug("已保存检测模型输出的概率图二值化后的图像, 文件名: bitmapImage.jpg, 文件路径: {}", debugPath);
    }

    public static void printContourImage(OCRContext context, String debugPath) {
        // 获取检测预处理图像
        Mat preDetMat = getPreDetImage(context);
        // 获取原始轮廓顶点
        List<ContourBox> contourBoxes = context.getDetState().getContourBoxes();
        // 原始轮廓框图
        Mat contourMat = preDetMat.clone();
        for (int i = 1; i <= contourBoxes.size(); i++) {
            ContourBox contourBox = contourBoxes.get(i - 1);
            // 绘制轮廓框 + 序号
            drawTextBox(contourMat,contourBox.getPoints(), String.valueOf(i));
        }
        // 保存
        OpenCVUtil.saveImage(contourMat, debugPath + "/contourImage.jpg");
        log.debug("已保存检测阶段的所有轮廓框图像, 文件名: contourImage.jpg, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(preDetMat);
        OpenCVUtil.releaseMat(contourMat);
    }

    public static void printDetBoxImage(OCRContext context, String debugPath) {
        // 获取检测还原后的顶点
        List<TextBox> detResultBoxes = context.getDetResultBoxes();
        // 绘制检测框
        Mat detMat = context.getRawMat().clone();
        detResultBoxes.forEach(box -> {
            // 绘制轮廓框: 序号 + 平均置信度 + 宽高比
            String text = box.getIndex() + ". score:" + BigDecimal.valueOf(box.getScore()).setScale(2, RoundingMode.HALF_UP)
                    + " aspectRatio:" + BigDecimal.valueOf(box.getAspectRatio()).setScale(2, RoundingMode.HALF_UP);
            drawTextBox(detMat, box.getPoints(), text);
        });
        // 保存
        OpenCVUtil.saveImage(detMat, debugPath + "/detBoxImage.jpg");
        log.debug("已保存检测阶段处理后的检测框图像, 文件名: detBoxImage.jpg, 文件路径: {}", debugPath);
        OpenCVUtil.releaseMat(detMat);
    }

    public static void printClsCropImage(OCRContext context, String debugPath) {
        // 分类检测裁剪文件夹
        String clsDebugPath = debugPath + "/clsCrop";
        ensureDir(clsDebugPath);
        // 获取原图和检测框
        Mat rawMat = context.getRawMat();
        List<ClsBatch> clsBatches = context.getClsBatches();
        // 创建裁剪图
        Mat rawCropMat = rawMat.clone();
        for (int i = 1; i <= clsBatches.size(); i++) {
            // 当前批次内检测框
            List<TextBox> batchBoxes = clsBatches.get(i - 1).getBoxes();
            // 当前批次模型输入尺寸
            Size modelInputSize = clsBatches.get(i - 1).getModelInputSize();
            // 当前批次裁剪文件夹
            String batchDebugPath = clsDebugPath + "/batch" + i;
            ensureDir(batchDebugPath);
            for (int j = 1; j <= batchBoxes.size(); j++) {
                // 当前检测框坐标
                Point[] points = batchBoxes.get(j - 1).getPoints();
                // 透视变换裁剪
                Mat cropMat = OpenCVUtil.perspectiveTransformCrop(rawCropMat, points);
                // 将已裁剪区域置空
                OpenCVUtil.fillPolyWhite(rawCropMat, points);
                // 缩放和转换转换RGB通道
                Mat rgbMat = OpenCVUtil.resizeToRGB(cropMat, modelInputSize);
                // 保存
                String cropImageName = "/crop" + j + ".jpg";
                String resizeImageName = "/resize" + j + ".jpg";
                OpenCVUtil.saveImage(cropMat, batchDebugPath + cropImageName);
                OpenCVUtil.saveImage(rgbMat, batchDebugPath + resizeImageName);
                log.trace("已保存当前检测框的分类检测阶段裁剪后的原图, 文件名: {}", cropImageName);
                log.trace("已保存当前检测框的分类检测阶段裁剪后的缩放图, 文件名: {}", resizeImageName);
                // 资源释放
                OpenCVUtil.releaseMat(cropMat);
                OpenCVUtil.releaseMat(rgbMat);
            }
            log.debug("已保存当前批次分类检测阶段裁剪图, 文件路径: {}", batchDebugPath);
        }
        log.debug("已保存分类检测阶段所有裁剪图, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
    }

    public static void printCropFillImage(OCRContext context, String debugPath) {
        // 获取原图和裁剪框坐标
        Mat rawMat = context.getRawMat();
        List<Point[]> boxes = context.getDetResultBoxes().stream().map(TextBox::getPoints).collect(Collectors.toList());
        // 创建裁剪图
        Mat rawCropMat = rawMat.clone();
        // 填充
        boxes.forEach(box -> {
            // 将裁剪区域置空
            OpenCVUtil.fillPolyWhite(rawCropMat, box);
            // 绘制裁剪框
            OpenCVUtil.drawBox(rawCropMat, box);
        });
        // 保存
        OpenCVUtil.saveImage(rawCropMat, debugPath + "/cropFillImage.jpg");
        log.debug("已保存裁剪后的效果填充图, 文件名: cropFillImage.jpg, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
    }

    public static void printClsResultImage(OCRContext context, String debugPath) {
        // 获取原图和裁剪框坐标
        Mat rawMat = context.getRawMat();
        List<TextBox> boxes = context.getClsResultBoxes();
        // 创建裁剪图
        Mat rawCropMat = rawMat.clone();
        // 填充
        boxes.forEach(box -> {
            // 填充角度和置信度
            String text =  "angle:" + box.getAngle() +
                    " confidence:" + BigDecimal.valueOf(box.getClsConfidence()).setScale(4, RoundingMode.HALF_UP);
            drawTextBox(rawCropMat, box.getPoints(), text);
        });
        // 保存
        OpenCVUtil.saveImage(rawCropMat, debugPath + "/clsResultImage.jpg");
        log.debug("已保存分类检测结果图, 文件名: clsResultImage.jpg, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
    }

    public static void printRotateImage(OCRContext context, String debugPath) {
        // 旋转检测框裁剪文件夹
        String rotateDebugPath = debugPath + "/rotateCrop";
        ensureDir(rotateDebugPath);
        // 裁剪图和绘制图
        Mat rawCropMat = context.getRawMat().clone();
        Mat drawBoxesMat = context.getRawMat().clone();
        // 获取需要旋转纠正的检测框
        List<TextBox> rotateBoxes = context.getRecBatches().
                stream().map(RecBatch::getBoxes).
                flatMap(List::stream).
                filter(TextBox::isRotate).
                collect(Collectors.toList());
        // 裁剪旋转框
        if (!rotateBoxes.isEmpty()) {
            for (int i = 1; i <= rotateBoxes.size(); i++) {
                // 当前检测框
                TextBox box = rotateBoxes.get(i - 1);
                // 绘制原图检测框
                // 填充角度和置信度
                String text =  "angle:" + box.getAngle() +
                        " confidence:" + BigDecimal.valueOf(box.getClsConfidence()).setScale(4, RoundingMode.HALF_UP);
                drawTextBox(drawBoxesMat, box.getPoints(), text);
                // 裁剪
                Mat cropMat = OpenCVUtil.perspectiveTransformCrop(rawCropMat, box.getPoints());
                // 保存
                String cropImageName = "/crop" + i + ".jpg";
                String rotateImageName = "/rotate" + i + ".jpg";
                OpenCVUtil.saveImage(cropMat, rotateDebugPath + cropImageName);
                log.trace("已保存需要旋转纠正的检测框裁剪图, 文件名: {}", cropImageName);
                // 旋转纠正
                OpenCVUtil.rotate(cropMat, box.getAngle());
                OpenCVUtil.saveImage(cropMat, rotateDebugPath + rotateImageName);
                log.trace("已保存旋转纠正后的检测框裁剪图, 文件名: {}", rotateImageName);
                // 资源释放
                OpenCVUtil.releaseMat(cropMat);
            }
            // 保存
            OpenCVUtil.saveImage(drawBoxesMat, debugPath + "/rotateImage.jpg");
            log.debug("已保存需要旋转纠正的检测框图, 文件名: rotateImage.jpg, 文件路径: {}", debugPath);
        }
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
        OpenCVUtil.releaseMat(drawBoxesMat);
    }

    public static void printRecCropImage(OCRContext context, String debugPath) {
        // 裁剪文件夹
        String recDebugPath = debugPath + "/recCrop";
        ensureDir(recDebugPath);
        // 获取原图和检测框
        Mat rawMat = context.getRawMat();
        List<RecBatch> recBatches = context.getRecBatches();
        // 创建裁剪图
        Mat rawCropMat = rawMat.clone();
        for (int i = 1; i <= recBatches.size(); i++) {
            // 当前批次内检测框
            List<TextBox> batchBoxes = recBatches.get(i - 1).getBoxes();
            // 当前批次模型输入尺寸
            Size modelInputSize = recBatches.get(i - 1).getModelInputSize();
            // 当前批次裁剪文件夹
            String batchDebugPath = recDebugPath + "/batch" + i;
            ensureDir(batchDebugPath);
            for (int j = 1; j <= batchBoxes.size(); j++) {
                // 当前检测框
                TextBox box = batchBoxes.get(j - 1);
                // 当前检测框坐标
                Point[] points = batchBoxes.get(j - 1).getPoints();
                // 透视变换裁剪
                Mat cropMat = OpenCVUtil.perspectiveTransformCrop(rawCropMat, points);
                // 将已裁剪区域置空
                OpenCVUtil.fillPolyWhite(rawCropMat, points);
                // 旋转纠正
                if (box.getAngle() != 0 && box.isRotate()) {
                    OpenCVUtil.rotate(cropMat, box.getAngle());
                }
                // 缩放到固定高度并转换通道
                Size fixHeightSize = OpenCVUtil.getFixHeightSize(cropMat.size(), (int) modelInputSize.height);
                Mat rgbMat = OpenCVUtil.resizeToRGB(cropMat, fixHeightSize);
                // 填充
                Mat paddingMat = OpenCVUtil.padding(rgbMat, modelInputSize);
                // 保存
                String cropImageName = "/crop" + j + ".jpg";
                String paddingImageName = "/padding" + j + ".jpg";
                OpenCVUtil.saveImage(cropMat, batchDebugPath + cropImageName);
                OpenCVUtil.saveImage(paddingMat, batchDebugPath + paddingImageName);
                log.trace("已保存当前检测框的图片识别阶段裁剪后的原图, 文件名: {}", cropImageName);
                log.trace("已保存当前检测框的图片识别阶段裁剪后的缩放填充图, 文件名: {}", paddingImageName);
                // 资源释放
                OpenCVUtil.releaseMat(cropMat);
                OpenCVUtil.releaseMat(rgbMat);
                OpenCVUtil.releaseMat(paddingMat);
            }
            log.debug("已保存当前批次图片识别阶段裁剪图, 文件路径: {}", batchDebugPath);
        }
        log.debug("已保存图片识别阶段所有裁剪图, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
    }

    public static void printRecConfidenceImage(OCRContext context, String debugPath) {
        // 获取原图和识别结果
        Mat rawMat = context.getRawMat();
        List<TextBox> confidenceBoxes = context.getRecBatches().stream().map(RecBatch::getBoxes).flatMap(List::stream).collect(Collectors.toList());
        // 创建裁剪图
        Mat rawCropMat = rawMat.clone();
        // 填充
        confidenceBoxes.forEach(box -> {
            String text =  "confidence:" + BigDecimal.valueOf(box.getRecConfidence()).setScale(4, RoundingMode.HALF_UP);
            drawTextBox(rawCropMat, box.getPoints(), text);
        });
        // 保存
        OpenCVUtil.saveImage(rawCropMat, debugPath + "/recConfidenceImage.jpg");
        log.debug("已保存图片识别置信度图, 文件名: recConfidenceImage.jpg, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
    }

    public static void printRecResultImage(OCRContext context, String debugPath) {
        // 获取原图和识别结果
        Mat rawMat = context.getRawMat();
        List<TextBox> boxes = context.getRecResultBoxes();
        // 创建裁剪图
        Mat rawCropMat = rawMat.clone();
        // 填充
        boxes.forEach(box -> {
            String text =  "text:" + box.getRecText() +
                    " confidence:" + BigDecimal.valueOf(box.getRecConfidence()).setScale(4, RoundingMode.HALF_UP);
            drawTextBox(rawCropMat, box.getPoints(), text);
        });
        // 保存
        OpenCVUtil.saveImage(rawCropMat, debugPath + "/recResultImage.jpg");
        log.debug("已保存图片识别结果图, 文件名: recResultImage.jpg, 文件路径: {}", debugPath);
        // 资源释放
        OpenCVUtil.releaseMat(rawCropMat);
    }

    private static void drawTextBox(Mat srcMat, Point[] points, String text) {
        // 填充识别结果和置信度
        OpenCVUtil.drawBox(srcMat, points);
        if (text != null && !text.isEmpty()) {
            OpenCVUtil.putText(srcMat, text, points);
        }
    }

    private static Mat getPreDetImage(OCRContext context){
        // 获取原图、缩放尺寸、填充尺寸
        Mat rawMat = context.getRawMat();
        Size resizeMatSize = context.getDetState().getResizeMatSize();
        Size modelInputSize = context.getDetState().getModelInputSize();
        // 缩放图像 + 转换RGB通道
        Mat rgbMat = OpenCVUtil.resizeToRGB(rawMat, resizeMatSize);
        // 填充
        return OpenCVUtil.padding(rgbMat, modelInputSize);
    }

    private static void safeRun(String name, Runnable task) {
        try {
            task.run();
        } catch (Exception e) {
            log.warn("Debug图片 {} 生成失败", name, e);
        }
    }

    private static void ensureDir(String dir) {
        File file = new File(dir);
        if (!file.exists() && !file.mkdirs()) {
            log.error("创建目录失败");
            throw new RuntimeException("Failed to create directory: " + dir);
        }
    }

}
