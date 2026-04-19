package com.ocr.paddleocr.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OCRConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    // ==================== 模型路径配置 ====================

    @Builder.Default
    private String modelDir = "src/main/resources/models";
    private String detModelPath;
    private String clsModelPath;
    private String recModelPath;
    private String dictPath;

    // ==================== 检测模型参数 ====================

    @Builder.Default
    private float detDbThresh = 0.3f;
    @Builder.Default
    private float detDbBoxThresh = 0.5f;
    @Builder.Default
    private float detDbUnclipRatio = 2.5f;
    @Builder.Default
    private float detDilationRatio = 2.0f;;
    @Builder.Default
    private int detMaxSideLen = 960;
    @Builder.Default
    private int detImageHeight = 960;
    @Builder.Default
    private int detImageWidth = 960;
    @Builder.Default
    private boolean detUsePolygon = false;

    // ==================== 识别模型参数 ====================

    @Builder.Default
    private int recImgH = 48;
    @Builder.Default
    private int recImgW = 320;
    @Builder.Default
    private int recBatchSize = 16;

    // ==================== 方向分类参数 ====================

    @Builder.Default
    private float clsThresh = 0.9f;
    @Builder.Default
    private boolean useAngleCls = true;

    // ==================== 系统参数 ====================

    @Builder.Default
    private boolean useGpu = false;
    @Builder.Default
    private int gpuId = 0;
    @Builder.Default
    private int numThreads = 4;
    @Builder.Default
    private boolean enableMKLDNN = false;

    // ==================== 后处理参数 ====================

    @Builder.Default
    private float boxThresh = 0.3f;
    @Builder.Default
    private float unclipRatio = 2.5f;
    @Builder.Default
    private float nmsIouThreshold = 0.5f;
    @Builder.Default
    private boolean dropScoreBelowThresh = true;

    // ==================== 语言配置 ====================

    @Builder.Default
    private String lang = "en";

    // ==================== 输出配置 ====================

    @Builder.Default
    private String outputDir = "output";

    /**
     * 是否保存可视化图片（对齐官方 visualize 参数）
     * 开启后会自动保存以下图片：
     * - 置信度热力图 (probability_map.jpg)
     * - 检测框图 (detection_boxes.jpg)
     * - 边缘检测图 (edge_map.jpg)
     * - 二值化图 (binary_map.jpg)
     * - 灰度图 (gray_map.jpg)
     * - 字符框图 (character_boxes.jpg)
     * - 裁剪图 (cropped/box_xxx.jpg)
     */
    @Builder.Default
    private boolean visualize = false;

    /**
     * 是否保存裁剪的文本框图片 - 当 visualize=true 时自动开启
     */
    @Builder.Default
    private boolean saveCroppedImages = false;

    // ==================== 过滤参数 ====================

    /**
     * 最小识别置信度，用于过滤低置信度文本内容
     */
    @Builder.Default
    private float minConfidence = 0.5f;
    @Builder.Default
    private int minTextLength = 1;
    @Builder.Default
    private int maxBoxWidth = 250;
    @Builder.Default
    private int minBoxArea = 9;
    @Builder.Default
    private int minBoxWidth = 5;
    @Builder.Default
    private int minBoxHeight = 5;
    @Builder.Default
    private float maxAspectRatio = 50.0f;

    // ==================== 预处理参数 ====================

    @Builder.Default
    private float[] mean = {0.485f, 0.456f, 0.406f};
    @Builder.Default
    private float[] std = {0.229f, 0.224f, 0.225f};
    @Builder.Default
    private int meanValue = 127;
    @Builder.Default
    private int scaleValue = 127;

    // ==================== 高级参数 ====================

    @Builder.Default
    private int maxCandidates = 1000;
    @Builder.Default
    private float mergeThreshold = 20f;
    @Builder.Default
    private int threadPoolSize = 4;
    @Builder.Default
    private boolean enableCache = false;
    @Builder.Default
    private int maxCacheSize = 100;

    // ==================== 便捷方法 ====================

    public String getDetModelPath() {
        if (detModelPath == null || detModelPath.isEmpty()) {
            return modelDir + "/det_model.onnx";
        }
        return detModelPath;
    }

    public String getClsModelPath() {
        if (clsModelPath == null || clsModelPath.isEmpty()) {
            return modelDir + "/cls_model.onnx";
        }
        return clsModelPath;
    }

    public String getRecModelPath() {
        if (recModelPath == null || recModelPath.isEmpty()) {
            return modelDir + "/rec_model.onnx";
        }
        return recModelPath;
    }

    public String getDictPath() {
        if (dictPath == null || dictPath.isEmpty()) {
            return modelDir + "/" + lang + "_dict.txt";
        }
        return dictPath;
    }

    public int getEffectiveDetImageHeight() {
        return detImageHeight > 0 ? detImageHeight : detMaxSideLen;
    }

    public int getEffectiveDetImageWidth() {
        return detImageWidth > 0 ? detImageWidth : detMaxSideLen;
    }

    /**
     * 设置可视化模式
     * 对齐官方 visualize 参数，开启后自动保存所有中间图片
     */
    public void setVisualize(boolean visualize) {
        this.visualize = visualize;
        if (visualize) {
            this.saveCroppedImages = true;
        }
    }

    public void validate() {
        if (detDbThresh <= 0 || detDbThresh >= 1) {
            throw new IllegalArgumentException("detDbThresh必须在0-1之间，当前值: " + detDbThresh);
        }
        if (detDbBoxThresh <= 0 || detDbBoxThresh >= 1) {
            throw new IllegalArgumentException("detDbBoxThresh必须在0-1之间，当前值: " + detDbBoxThresh);
        }
        if (detDbUnclipRatio <= 0) {
            throw new IllegalArgumentException("detDbUnclipRatio必须大于0，当前值: " + detDbUnclipRatio);
        }
        if (recBatchSize <= 0 || recBatchSize > 64) {
            throw new IllegalArgumentException("recBatchSize必须在1-64之间，当前值: " + recBatchSize);
        }
        if (threadPoolSize <= 0 || threadPoolSize > 32) {
            throw new IllegalArgumentException("threadPoolSize必须在1-32之间，当前值: " + threadPoolSize);
        }
        if (minConfidence < 0 || minConfidence > 1) {
            throw new IllegalArgumentException("minConfidence必须在0-1之间，当前值: " + minConfidence);
        }
        if (lang == null || lang.isEmpty()) {
            throw new IllegalArgumentException("lang不能为空");
        }
    }

    // ==================== 预设配置 ====================

    public static OCRConfig defaultEnglish() {
        return OCRConfig.builder()
                .lang("en")
                .build();
    }

    public static OCRConfig defaultChinese() {
        return OCRConfig.builder()
                .lang("ch")
                .detDbUnclipRatio(1.5f)
                .build();
    }

    public static OCRConfig highPerformance() {
        return OCRConfig.builder()
                .useGpu(true)
                .gpuId(0)
                .numThreads(8)
                .recBatchSize(32)
                .threadPoolSize(8)
                .build();
    }

    public static OCRConfig lowMemory() {
        return OCRConfig.builder()
                .useGpu(false)
                .numThreads(2)
                .recBatchSize(4)
                .threadPoolSize(2)
                .detMaxSideLen(640)
                .build();
    }

    public static OCRConfig highAccuracy() {
        return OCRConfig.builder()
                .detDbThresh(0.2f)
                .detDbBoxThresh(0.3f)
                .minConfidence(0.3f)
                .useAngleCls(true)
                .visualize(true)
                .build();
    }

    @Override
    public String toString() {
        return String.format("OCRConfig{lang='%s', useGpu=%s, detDbThresh=%.2f, " +
                        "detDbUnclipRatio=%.1f, recBatchSize=%d, visualize=%s}",
                lang, useGpu, detDbThresh, detDbUnclipRatio, recBatchSize, visualize);
    }
}