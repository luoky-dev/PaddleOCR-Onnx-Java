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

    // ==================== 模型基础配置 ====================

    @Builder.Default
    private String detModelPath = "";
    @Builder.Default
    private String clsModelPath = "";
    @Builder.Default
    private String recModelPath = "";
    @Builder.Default
    private String dictPath = "";
    @Builder.Default
    private String debugPath = "";
    // 是否启用分类检测模型
    @Builder.Default
    private boolean useCls = false;
    // 是否启用debug模式
    @Builder.Default
    private boolean useDebug = false;
    // 分批处理大小
    @Builder.Default
    private int batchSize = 6;

    // ==================== 检测模型参数 ====================

    // 二值化阈值
    @Builder.Default
    private float bitThresh = 0.6f;
    // 是否使用膨胀
    @Builder.Default
    private boolean isDilation = false;
    // 腐蚀度
    @Builder.Default
    private final float epsilon = 0.002f;
    // unclip 扩张比率
    @Builder.Default
    private float unclipRatio = 1.3f;
    // 检测框数量限制
    @Builder.Default
    private int boxLimit = 1000;
    // 检测框最小尺寸过滤
    @Builder.Default
    private int boxMinSize = 5;
    // 检测框最小面积过滤
    @Builder.Default
    private int boxMinArea = 20;
    // 检测框最小宽高比阈值(小于阈值执行旋转操作)
    @Builder.Default
    private float boxMinAspectRatio = 0.67f;
    // 检测框最低置信度阈值过滤
    @Builder.Default
    private float boxThresh = 0.6f;

    // ==================== 方向分类参数 ====================

    // 分类检测阈值
    @Builder.Default
    private float clsThresh = 0.9f;

    // ==================== 图像识别参数 ====================

    // 图像识别最低阈值
    @Builder.Default
    private float recThresh = 0.7f;

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
        validatePositive("detMinSize", boxMinSize, errors);           // 最小检测框尺寸
        validatePositive("batchSize", batchSize, errors);             // 批量处理大小
        validatePositive("numThreads", numThreads, errors);           // 线程数
        validatePositive("boxLimit", boxLimit, errors);           // 检测框数量限制

        // 范围验证 [0, 1]
        validateRange01("detThresh", bitThresh, errors);              // 检测阈值
        validateRange01("detBoxThresh", boxThresh, errors);        // 检测框置信度阈值
        validateRange01("clsThresh", clsThresh, errors);              // 分类置信度阈值

        // 正有限数验证
        validateFinitePositive("detUnclipRatio", unclipRatio, errors);  // Unclip扩张比例
        validateFinitePositive("epsilon", epsilon, errors);  // 腐蚀度

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
