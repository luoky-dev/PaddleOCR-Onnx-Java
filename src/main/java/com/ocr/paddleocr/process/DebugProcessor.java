package com.ocr.paddleocr.process;

import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

@Slf4j
public class DebugProcessor {

    public static void printDebugImages(OCRContext context, OCRConfig config, String debugPath) {
        if (context == null || debugPath == null || debugPath.trim().isEmpty()) {
            return;
        }
        OpenCVUtil.ensureDir(debugPath);

        safeRun("det_boxes", () -> printDetBoxesImage(context, debugPath));
        safeRun("det_prep", () -> printDetPrepImage(context, debugPath));
        safeRun("det_prob_heatmap", () -> printDetProbHeatmapImage(context, debugPath));
        safeRun("det_binary_map", () -> printDetBinaryMapImage(context, config, debugPath));
        safeRun("det_contours", () -> printContourImages(context, debugPath));
        safeRun("det_crops", () -> printDetCropImages(context, debugPath));
        safeRun("cls_compare", () -> printClsCompareImages(context, debugPath));
        safeRun("rec_overlay", () -> printRecOverlayImage(context, debugPath));
    }

    /**
     * 在原图上绘制检测框
     * 用途：查看检测模型找到的文本区域位置
     */
    public static void printDetBoxesImage(OCRContext context, String debugPath) {
        // 收集所有检测框的顶点坐标
        List<List<Point>> detectBoxes = new ArrayList<>();
        if (context.getDetResultBoxes() != null) {
            context.getDetResultBoxes().forEach(box -> detectBoxes.add(box.getRestorePoints()));
        }
        // 在原图上绘制
        Mat detectImage = OpenCVUtil.drawBoxes(context.getRawMat(), detectBoxes);
        OpenCVUtil.saveImageAndRelease(detectImage,debugPath + "/det_boxes.jpg");
        log.debug("已保存检测框图, 文件名: det_boxes.jpg, 文件路径: {}", debugPath);
    }

    /**
     * 检测预处理后的图像
     * 用途：查看输入检测模型的图像（缩放、归一化后的效果）
     */
    public static void printDetPrepImage(OCRContext context, String debugPath) {
        // 将预处理图像转换为可视化的8UC3格式
        Mat vis = OpenCVUtil.toVisualizableImage(context.getDetPrepMat(), true);
        OpenCVUtil.saveImageAndRelease(vis,debugPath + "/det_prep.jpg");
        log.debug("已保存检测预处理后的图像, 文件名: det_prep.jpg, 文件路径: {}", debugPath);
    }

    /**
     * 检测模型输出的概率热力图
     * 用途：查看模型对文本区域的预测置信度分布
     * - 红色/黄色：高概率（文本区域）
     * - 蓝色/黑色：低概率（背景）
     */
    public static void printDetProbHeatmapImage(OCRContext context, String debugPath) {
        // 将概率图转换为伪彩色热力图
        Mat heatmap = OpenCVUtil.createProbHeatmap(context.getDetProbMap());
        OpenCVUtil.saveImageAndRelease(heatmap, debugPath + "/det_prob_heatmap.jpg");
        log.debug("已保存检测模型输出的概率热力图, 文件名: det_prob_heatmap.jpg, 文件路径: {}", debugPath);
    }

    /**
     * 概率图二值化后的图像
     * 用途：查看经过阈值过滤后的文本区域
     * - 白色：超过阈值的区域（可能是文本）
     * - 黑色：低于阈值的区域（背景）
     */
    public static void printDetBinaryMapImage(OCRContext context, OCRConfig config, String debugPath) {
        // 获取阈值（默认0.3）
        float threshold = config == null ? 0.3f : config.getDetThresh();
        // 创建二值图（大于阈值=255，否则=0）
        Mat binary = OpenCVUtil.createBinaryMap(context.getDetProbMap(), threshold);
        OpenCVUtil.saveImageAndRelease(binary, debugPath + "/det_binary_map.jpg");
        log.debug("已保存检测模型输出的概率图二值化后的图像, 文件名: det_binary_map.jpg, 文件路径: {}", debugPath);
    }

    /**
     * 绘制轮廓对比图
     * 用途：对比原始轮廓和还原后的轮廓
     * - 蓝色：缩放图上的轮廓
     * - 红色：还原到原图的轮廓
     */
    public static void printContourImages(OCRContext context, String debugPath) {
        // 收集轮廓点
        List<List<Point>> contourPoints = new ArrayList<>();
        List<List<Point>> restorePoints = new ArrayList<>();
        if (context.getDetResultBoxes() != null) {
            for (TextBox box : context.getDetResultBoxes()) {
                if (box == null) {
                    continue;
                }
                // 原轮廓点
                contourPoints.add(box.getContourPoint());
                // 还原后的点
                restorePoints.add(box.getRestorePoints());
            }
        }
        // 在预处理图上绘制
        Mat prepVis = OpenCVUtil.toVisualizableImage(context.getDetPrepMat(), true);
        Mat prepContour = OpenCVUtil.drawBoxes(prepVis, contourPoints);
        OpenCVUtil.saveImageAndRelease(prepContour, debugPath + "/det_crops_prep.jpg");
        log.debug("已保存原始轮廓图像, 文件名: det_crops_prep.jpg, 文件路径: {}", debugPath);
        OpenCVUtil.releaseMat(prepVis);
        // 在原图上绘制
        Mat restoreContour = OpenCVUtil.drawBoxes(context.getRawMat(), restorePoints);
        OpenCVUtil.saveImageAndRelease(restoreContour, debugPath + "/det_crops_restore.jpg");
        log.debug("已保存还原后的轮廓图像, 文件名: det_crops_restore.jpg, 文件路径: {}", debugPath);
    }

    /**
     * 保存每个检测框的裁剪图像
     * 用途：查看每个文本区域的原始图像（用于分类和识别）
     */
    public static void printDetCropImages(OCRContext context, String debugPath) {
        if (context.getDetResultBoxes() == null || context.getDetResultBoxes().isEmpty()) {
            return;
        }
        // 创建子目录
        String cropDir = debugPath + "/det_crops";
        OpenCVUtil.ensureDir(cropDir);
        // 逐个保存裁剪图
        int idx = 0;
        for (TextBox box : context.getDetResultBoxes()) {
            if (box == null || box.getRestoreMat() == null || box.getRestoreMat().empty()) {
                continue;
            }
            String file = String.format(Locale.ROOT, "%s/det_crop_%03d.jpg", cropDir, idx++);
            OpenCVUtil.saveImage(box.getRestoreMat(), file);
        }
        log.debug("已保存每个检测框的裁剪图像, 裁剪文件个数: {}, 文件路径: {}", context.getDetResultBoxes().size(), cropDir);
    }

    /**
     * 生成分类（旋转校正）前后的对比图
     * 用途：查看角度分类和旋转校正的效果
     * - 左图：原始裁剪图
     * - 右图：旋转校正后的图
     */
    public static void printClsCompareImages(OCRContext context, String debugPath) {
        if (context.getClsResultBoxes() == null || context.getClsResultBoxes().isEmpty()) {
            return;
        }
        String compareDir = debugPath + "/cls_compare";
        OpenCVUtil.ensureDir(compareDir);

        int idx = 0;
        for (TextBox box : context.getClsResultBoxes()) {
            if (box == null || box.getRestoreMat() == null || box.getRestoreMat().empty()) {
                continue;
            }
            // 原始图像（旋转前）
            Mat before = OpenCVUtil.toVisualizableImage(box.getRestoreMat(), false);
            // 旋转后图像（如果有旋转）
            Mat after = OpenCVUtil.toVisualizableImage(box.getRotMat(), false);
            if (before.empty() || after.empty()) {
                OpenCVUtil.releaseMat(before);
                OpenCVUtil.releaseMat(after);
                continue;
            }
            // 统一高度后左右拼接
            if (before.rows() != after.rows()) {
                Mat resizedAfter = OpenCVUtil.resizeToHeight(after, before.rows());
                OpenCVUtil.releaseMat(after);
                after = resizedAfter;
            }
            // 左右拼接对比图
            Mat merged = OpenCVUtil.concatHorizontal(List.of(before, after));
            // 添加文字标注
            String text = String.format(
                    Locale.ROOT,
                    "angle=%d conf=%.3f rotated=%s",
                    box.getAngle(),
                    box.getClsConfidence(),
                    box.isRotate());
            OpenCVUtil.drawText(merged, text, new Point(10, 20));

            String file = String.format(Locale.ROOT, "%s/cls_compare_%03d.jpg", compareDir, idx++);
            OpenCVUtil.saveImageAndRelease(merged, file);
            OpenCVUtil.releaseMat(before);
            OpenCVUtil.releaseMat(after);
        }
        log.debug("已保存分类（旋转校正）前后的对比图, 裁剪文件个数: {}, 文件路径: {}", context.getClsResultBoxes().size(), compareDir);
    }

    /**
     * 在原图上叠加识别结果
     * 用途：查看最终的OCR效果
     * - 绿色框：文本区域
     * - 红色文字：识别出的文本和置信度
     */
    public static void printRecOverlayImage(OCRContext context, String debugPath) {
        if (context.getRawMat() == null || context.getRawMat().empty()) {
            return;
        }
        // 克隆原图（避免修改原图）
        Mat overlay = context.getRawMat().clone();
        if (context.getRecResultBoxes() != null) {
            for (TextBox box : context.getRecResultBoxes()) {
                if (box == null || box.getRestorePoints() == null || box.getRestorePoints().size() < 3) {
                    continue;
                }
                // 绘制检测框
                OpenCVUtil.drawPolygon(overlay, box.getRestorePoints());
                // 准备文字标签
                String recText = Objects.toString(box.getRecText(), "");
                if (recText.length() > 24) {
                    recText = recText.substring(0, 24) + "...";
                }
                String label = String.format(Locale.ROOT, "%s(%.3f)", recText, box.getRecConfidence());
                // 在文本框左上角添加文字
                Point anchor = box.getRestorePoints().get(0);
                OpenCVUtil.drawText(overlay, label, new Point(anchor.x, Math.max(15, anchor.y - 3)));
            }
        }
        OpenCVUtil.saveImageAndRelease(overlay, debugPath + "/rec_overlay.jpg");
        log.debug("已保存叠加识别的结果图, 文件名: rec_overlay.jpg, 文件路径: {}", debugPath);
    }

    private static void safeRun(String name, Runnable task) {
        try {
            task.run();
        } catch (Exception e) {
            log.warn("Debug图片 {} 生成失败", name, e);
        }
    }

}
