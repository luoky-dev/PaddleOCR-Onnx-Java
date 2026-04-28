package com.ocr.paddleocr.service.Impl;

import com.google.gson.Gson;
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.domain.OCRContext;
import com.ocr.paddleocr.domain.OCRResult;
import com.ocr.paddleocr.domain.TextBox;
import com.ocr.paddleocr.domain.Word;
import com.ocr.paddleocr.process.ClsProcessor;
import com.ocr.paddleocr.process.DebugProcessor;
import com.ocr.paddleocr.process.DetProcessor;
import com.ocr.paddleocr.process.ModelManager;
import com.ocr.paddleocr.process.RecProcessor;
import com.ocr.paddleocr.utils.OpenCVUtil;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * OCR服务实现类 - 单例模式
 * 负责具体的OCR识别逻辑
 */
@Slf4j
public class OCRServiceImpl {

    private static volatile OCRServiceImpl instance;
    private static volatile OCRServiceImpl customInstance;
    private final Gson gson;
    private final ModelManager modelManager;
    private final DetProcessor detProcessor;
    private final ClsProcessor clsProcessor;
    private final RecProcessor recProcessor;
    private final OCRConfig ocrConfig;
    private volatile boolean initialized;

    /**
     * 私有构造 - 使用默认配置
     */
    private OCRServiceImpl() {
        this(OCRConfig.builder().build());
    }

    /**
     * 私有构造 - 使用自定义配置
     */
    private OCRServiceImpl(OCRConfig ocrConfig) {
        if (ocrConfig == null) {
            throw new IllegalArgumentException("OCR服务配置不能为空");
        }
        this.gson = new Gson();
        try {
            ocrConfig.validate();
            this.ocrConfig = ocrConfig;
            this.modelManager = ModelManager.getInstance();
            synchronized (modelManager) {
                if (!modelManager.isInitialized()) {
                    modelManager.init(ocrConfig);
                }
            }
            this.detProcessor = new DetProcessor(modelManager);
            this.clsProcessor = new ClsProcessor(modelManager);
            this.recProcessor = new RecProcessor(modelManager);
            this.initialized = true;
            log.info("OCR服务初始化完成");
        } catch (Exception e) {
            log.error("OCR服务实现初始化失败", e);
            throw new RuntimeException("OCR服务实现初始化失败", e);
        }
    }

    /**
     * 获取单例实例（使用默认配置）
     */
    public static OCRServiceImpl getInstance() {
        if (instance == null) {
            synchronized (OCRServiceImpl.class) {
                if (instance == null) {
                    instance = new OCRServiceImpl();
                }
            }
        }
        return instance;
    }

    /**
     * 获取单例实例（使用自定义配置）
     */
    public static OCRServiceImpl getInstance(OCRConfig config) {
        if (customInstance == null) {
            synchronized (OCRServiceImpl.class) {
                if (customInstance == null) {
                    customInstance = new OCRServiceImpl(config);
                } else {
                    log.warn("OCRServiceImpl已使用自定义配置初始化, 新配置将被忽略");
                }
            }
        }
        return customInstance;
    }

    /**
     * 识别图片并返回JSON字符串
     *
     * @param imagePath 图片路径
     * @return JSON格式的识别结果
     */
    public String recognize(String imagePath) {
        OCRResult result = rec(imagePath);
        return gson.toJson(result);
    }

    /**
     * 识别图片并返回OCRResult对象
     *
     * @param imagePath 图片路径
     * @return OCRResult对象
     */
    private OCRResult rec(String imagePath) {
        OCRResult.OCRResultBuilder builder = OCRResult.builder()
                .imagePath(imagePath)
                .success(Boolean.FALSE);

        if (!initialized) {
            return builder.error("OCR服务未初始化").build();
        }
        OCRContext context = new OCRContext();
        long startTime = System.currentTimeMillis();
        try {
            // 读取图片
            context.setRawMat(OpenCVUtil.getImage(imagePath));
            log.info("开始图像识别");
            // 图像检测和切割
            detProcessor.detect(context);
            if (context.getDetResultBoxes().isEmpty()){
                return builder
                        .success(Boolean.FALSE)
                        .error("未检测到文本框, 识别失败")
                        .imageWidth(context.getRawMat().width())
                        .imageHeight(context.getRawMat().height())
                        .processingTime(System.currentTimeMillis() - startTime)
                        .build();
            }
            log.info("检测完成, 检测框数量: {}, 检测处理时间: {} ms",
                    context.getDetResultBoxes().size(),
                    context.getDetProcessTime());
            // 启用分类检测时进行分类检测和纠正
            log.info("开始角度分类处理, 文本框数量: {}", context.getDetResultBoxes().size());
            if (ocrConfig.isUseCls()) {
                clsProcessor.classify(context);
                log.info("方向分类已启用, 倾斜框纠正数量: {}, 分类检测处理时间: {} ms",
                        context.getClsResultBoxes().stream().filter(TextBox::isRotate).count(),
                        context.getClsProcessTime()
                );
            } else {
                log.info("方向分类未启用, 将跳过方向分类使用检测模型结果进行识别");
            }
            // 检测框识别
            recProcessor.recognize(context);
            log.info("检测框识别完成, 成功识别检测框数量: {}, 识别时间: {} ms",
                    context.getRecResultBoxes().size(),
                    context.getRecProcessTime());
            if (ocrConfig.isUseDebug()) {
                log.info("Debug模式已启用, 打印中间图像信息到 {} 目录", ocrConfig.getDebugPath());
                DebugProcessor.printDebugImages(context, ocrConfig, ocrConfig.getDebugPath());
            }
            if (context.getRecResultBoxes().isEmpty()){
                return builder
                        .success(Boolean.FALSE)
                        .error("无文本框识别结果, 识别失败")
                        .imageWidth(context.getRawMat().width())
                        .imageHeight(context.getRawMat().height())
                        .processingTime(System.currentTimeMillis() - startTime)
                        .build();
            }
            List<Word> words = new ArrayList<>();
            context.getRecResultBoxes().forEach(textBox -> words.add(Word.builder()
                    .text(textBox.getRecText())
                    .confidence(textBox.getRecConfidence())
                    .box(textBox.getRestorePoints())
                    .build()));

            return builder
                    .success(Boolean.TRUE)
                    .words(words)
                    .imageWidth(context.getRawMat().width())
                    .imageHeight(context.getRawMat().height())
                    .processingTime(System.currentTimeMillis() - startTime)
                    .build();
        } catch (Exception e) {
            log.error("OCR识别失败: {}", imagePath, e);
            return builder.error(e.getMessage()).build();
        } finally {
            releaseResources(context);
        }
    }

    /**
     * 释放 OCRContext/TextBox 中持有的所有本地资源
     */
    private void releaseResources(OCRContext context) {
        if (context == null) {
            return;
        }

        try {
            Set<TextBox> visited = Collections.newSetFromMap(new IdentityHashMap<>());
            releaseTextBoxes(context.getDetResultBoxes(), visited);
            releaseTextBoxes(context.getClsResultBoxes(), visited);
            releaseTextBoxes(context.getRecResultBoxes(), visited);

            OpenCVUtil.releaseMat(context.getDetPrepMat());
            OpenCVUtil.releaseMat(context.getRawMat());
        } catch (Exception e) {
            log.warn("释放OCR上下文资源失败", e);
        } finally {
            context.setRawMat(null);
            context.setDetPrepMat(null);
            context.setDetProbMap(null);
            context.setDetResultBoxes(null);
            context.setClsBatchBoxes(null);
            context.setClsBatchChw(null);
            context.setClsLogitsList(null);
            context.setClsResultBoxes(null);
            context.setRecResultBoxes(null);
        }
    }

    private void releaseTextBoxes(List<TextBox> boxes, Set<TextBox> visited) {
        if (boxes == null) {
            return;
        }
        for (TextBox box : boxes) {
            if (box == null || !visited.add(box)) {
                continue;
            }
            OpenCVUtil.releaseMat(box.getRestoreMat());
            OpenCVUtil.releaseMat(box.getRotMat());
            box.setContourPoint(null);
            box.setRestoreMat(null);
            box.setRestorePoints(null);
            box.setRotMat(null);
        }
    }

    public synchronized void restart() {
        shutdown();
        try {
            synchronized (modelManager) {
                if (!modelManager.isInitialized()) {
                    modelManager.init(ocrConfig);
                }
            }
            initialized = true;
            log.info("OCR服务已重启");
        } catch (Exception e) {
            log.error("OCR服务重启失败", e);
            throw new RuntimeException("OCR服务重启失败", e);
        }
    }

    public void shutdown() {
        if (modelManager != null) {
            modelManager.close();
        }
        initialized = false;
        log.info("OCR服务已关闭");
    }
}
