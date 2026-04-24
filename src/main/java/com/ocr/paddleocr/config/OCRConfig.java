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
    private String detModelPath = "src/main/java/resources/models/chi/det_model.onnx";
    @Builder.Default
    private String clsModelPath = "src/main/java/resources/models/chi/cls_model.onnx";
    @Builder.Default
    private String recModelPath = "src/main/java/resources/models/chi/rec_model.onnx";
    @Builder.Default
    private String dictPath = "src/main/java/resources/models/chi/ppocr_keys_v1.txt";
    @Builder.Default
    private String debugPath = "src/main/java/resources/test/output";
    // 是否启用分类检测模型
    @Builder.Default
    private boolean useCls = true;
    // 是否启用debug模式
    @Builder.Default
    private boolean useDebug = true;
    // ==================== 检测模型参数 ====================

    // 识别模型固态输入高度
    @Builder.Default
    private int detModelHeight = 960;
    // 识别模型固态输入宽度
    @Builder.Default
    private int detModelWidth = 960;
    // 二值化阈值
    @Builder.Default
    private float detThresh = 0.3f;
    // 是否使用膨胀
    @Builder.Default
    private boolean isDilation = true;
    // 框置信度阈值过滤
    @Builder.Default
    private float detBoxThresh = 0.5f;
    // unclip 扩张比率
    @Builder.Default
    private float detUnclipRatio = 1.6f;
    // 最小检测框尺寸过滤（px）
    @Builder.Default
    private int detMinSize = 5;
    // 是否返回多边形（false返回矩形）
    @Builder.Default
    private boolean detUsePolygon = true;

    // ==================== 方向分类参数 ====================

    // 分类检测阈值
    @Builder.Default
    private float clsThresh = 0.9f;
    // 检测框分批处理大小
    @Builder.Default
    private int batchSize = 64;

    // ==================== 系统参数 ====================

    @Builder.Default
    private boolean useGpu = false;
    @Builder.Default
    private int gpuId = 0;
    @Builder.Default
    private int numThreads = 4;
    @Builder.Default
    private boolean enableMKLDNN = false;

}