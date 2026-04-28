package com.ocr.paddleocr.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

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
    private boolean useCls = false;
    // 是否启用debug模式
    @Builder.Default
    private boolean useDebug = false;
    // ==================== 检测模型参数 ====================

    // 是否使用启用预检测(适用图片范围: 大图密集文字、大量白边、大量无有效文本区域)
    @Builder.Default
    private boolean usePreDet = false;
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
    private float detUnclipRatio = 1.3f;
    // 最小检测框尺寸过滤（px）
    @Builder.Default
    private int detMinSize = 5;
    // 是否返回多边形（false返回矩形）
    @Builder.Default
    private boolean detUsePolygon = false;

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
    private int numThreads = 8;

    /**
     * 验证OCR配置的完整性和有效性
     * 在OCR服务初始化前调用，确保所有配置参数合法
     *
     * @throws IllegalArgumentException 当任何验证失败时抛出，包含所有错误信息
     */
    public void validate() {
        // 收集所有错误信息，一次性返回给调用者
        List<String> errors = new ArrayList<>();

        // 必需文件验证
        // 检测模型文件必须存在且可读
        validateRequiredFile("detModelPath", detModelPath, errors);
        // 识别模型文件必须存在且可读
        validateRequiredFile("recModelPath", recModelPath, errors);
        // 字典文件必须存在且可读
        validateRequiredFile("dictPath", dictPath, errors);

        // 如果启用了角度分类，分类模型文件也必须存在
        if (useCls) {
            validateRequiredFile("clsModelPath", clsModelPath, errors);
        }

        // 调试路径验证
        if (useDebug) {
            if (isBlank(debugPath)) {
                errors.add("debugPath must not be blank when useDebug=true");
            } else {
                File dir = new File(debugPath);
                // 检查路径是否存在且是否为目录
                if (dir.exists() && !dir.isDirectory()) {
                    errors.add("debugPath is not a directory: " + debugPath);
                }
                // 如果目录不存在，尝试创建
                else if (!dir.exists() && !dir.mkdirs()) {
                    errors.add("failed to create debugPath directory: " + debugPath);
                }
            }
        }

        // 数值参数验证
        // 正整数验证
        validatePositive("detModelHeight", detModelHeight, errors);   // 检测模型高度
        validatePositive("detModelWidth", detModelWidth, errors);     // 检测模型宽度
        validatePositive("detMinSize", detMinSize, errors);           // 最小文本框尺寸
        validatePositive("batchSize", batchSize, errors);             // 批量处理大小
        validatePositive("numThreads", numThreads, errors);           // 线程数

        // 范围验证 [0, 1]
        validateRange01("detThresh", detThresh, errors);              // 检测阈值
        validateRange01("detBoxThresh", detBoxThresh, errors);        // 检测框置信度阈值
        validateRange01("clsThresh", clsThresh, errors);              // 分类置信度阈值

        // 正有限数验证
        validateFinitePositive("detUnclipRatio", detUnclipRatio, errors);  // Unclip扩张比例

        // GPU参数验证
        if (gpuId < 0) {
            errors.add("gpuId must be >= 0");
        }

        // 抛出验证异常
        if (!errors.isEmpty()) {
            throw new IllegalArgumentException("Invalid OCRConfig: " + String.join("; ", errors));
        }
    }

    /**
     * 验证必需文件是否存在且可读
     *
     * @param field 字段名称（用于错误消息）
     * @param path 文件路径
     * @param errors 错误列表
     */
    private static void validateRequiredFile(String field, String path, List<String> errors) {
        // 1. 检查路径是否为空
        if (isBlank(path)) {
            errors.add(field + " must not be blank");
            return;
        }

        File file = new File(path);

        // 2. 检查文件是否存在
        if (!file.exists()) {
            errors.add(field + " file does not exist: " + path);
            return;
        }

        // 3. 检查是否为文件（而不是目录）
        if (!file.isFile()) {
            errors.add(field + " is not a file: " + path);
            return;
        }

        // 4. 检查文件是否可读
        if (!file.canRead()) {
            errors.add(field + " is not readable: " + path);
        }
    }

    /**
     * 验证正整数
     * 用于：尺寸、批次大小、线程数等
     */
    private static void validatePositive(String field, int value, List<String> errors) {
        if (value <= 0) {
            errors.add(field + " must be > 0");
        }
    }

    /**
     * 验证范围在 [0, 1] 之间的浮点数
     * 用于：各种阈值参数（概率值）
     */
    private static void validateRange01(String field, float value, List<String> errors) {
        // Float.isFinite() 检查是否为有效数值（非无穷大、非NaN）
        if (!Float.isFinite(value) || value < 0.0f || value > 1.0f) {
            errors.add(field + " must be in [0, 1]");
        }
    }

    /**
     * 验证正有限浮点数
     * 用于：扩张比例等必须为正数的参数
     */
    private static void validateFinitePositive(String field, float value, List<String> errors) {
        if (!Float.isFinite(value) || value <= 0.0f) {
            errors.add(field + " must be finite and > 0");
        }
    }

    /**
     * 检查字符串是否为空白
     *
     * @param text 待检查的字符串
     * @return true 如果为 null、空字符串或仅包含空白字符
     */
    private static boolean isBlank(String text) {
        return text == null || text.trim().isEmpty();
    }
}
