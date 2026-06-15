package com.ocr.paddleocr.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OCRConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    // ==================== 模型基础配置 ====================

    @Builder.Default
    private String detModelPath = null;
    @Builder.Default
    private String clsModelPath = null;
    @Builder.Default
    private String recModelPath = null;
    @Builder.Default
    private String dictPath = null;
    @Builder.Default
    private String debugPath = null;
    // 是否启用分类检测模型
    @Builder.Default
    private boolean useCls = false;
    // 是否启用debug模式
    @Builder.Default
    private boolean useDebug = false;
    // 分批处理大小
    @Builder.Default
    private int batchSize = 6;

    // ==================== 模型参数配置 ====================

    // 对齐倍数 (paddleOCR官方要求 32 的倍数) 
    @Builder.Default
    private int stride = 32;

    // 减均值除方差 (Z-score Normalization) - ImageNet RGB均值
    @Builder.Default
    private float[] scoreMean = {0.485f, 0.456f, 0.406f};

    // 减均值除方差 (Z-score Normalization) - ImageNet RGB标准差
    @Builder.Default
    private float[] scoreStd = {0.229f, 0.224f, 0.225f};

    // 线性 (Linear Scaling) - 线性到 [-1,1] RGB均值
    @Builder.Default
    private float[] linearMean = {0.5f, 0.5f, 0.5f};

    // 线性 (Linear Scaling) - 线性到 [-1,1] RGB标准差
    @Builder.Default
    private float[] linearStd = {0.5f, 0.5f, 0.5f};

    // 膨胀核大小 (paddleOCR官方默认 3) 
    @Builder.Default
    private int dilateKernelSize = 3;

    // 方向分类模型输入宽度
    @Builder.Default
    private int clsModelWidth = 320;

    // 方向分类模型输入高度
    @Builder.Default
    private int clsModelHeight = 48;

    // 角度分类字典
    @Builder.Default
    private int[] angleDict = {0, 180};

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
    private float unclipRatio = 1.6f;
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
     * 在OCR服务初始化前调用, 确保所有配置参数合法
     *
     * @throws IllegalArgumentException 当任何验证失败时抛出, 包含所有错误信息
     */
    public void validate() {
        // 收集所有错误信息, 一次性返回给调用者
        List<String> errors = new ArrayList<>();

        // ==================== 必需文件验证 ====================
        validateRequiredFile("detModelPath", detModelPath, errors);
        validateRequiredFile("recModelPath", recModelPath, errors);
        validateRequiredFile("dictPath", dictPath, errors);

        // 如果启用了角度分类, 分类模型文件也必须存在
        if (useCls) {
            validateRequiredFile("clsModelPath", clsModelPath, errors);
        }

        // ==================== 调试路径验证 ====================
        if (useDebug) {
            validateDebugPath(debugPath, errors);
        }

        // ==================== 整数参数验证 ====================
        validatePositive("stride", stride, errors);
        validatePositive("batchSize", batchSize, errors);
        validatePositive("numThreads", numThreads, errors);
        validatePositive("boxLimit", boxLimit, errors);
        validatePositive("boxMinSize", boxMinSize, errors);
        validatePositive("boxMinArea", boxMinArea, errors);
        validatePositive("dilateKernelSize", dilateKernelSize, errors);
        validatePositive("clsModelWidth", clsModelWidth, errors);
        validatePositive("clsModelHeight", clsModelHeight, errors);

        // stride 必须是32的倍数 (PaddleOCR官方要求) 
        if (stride % 32 != 0) {
            errors.add("stride must be a multiple of 32 (PaddleOCR requirement)");
        }

        // ==================== 浮点数参数验证 ====================
        validateRange01("bitThresh", bitThresh, errors);
        validateRange01("boxThresh", boxThresh, errors);
        validateRange01("clsThresh", clsThresh, errors);
        validateRange01("recThresh", recThresh, errors);
        validateRangePositive("unclipRatio", unclipRatio, errors);
        validateRangePositive("epsilon", epsilon, errors);
        validateRangePositive("boxMinAspectRatio", boxMinAspectRatio, errors);

        // ==================== 数组参数验证 ====================
        validateMeanStdArray("scoreMean", scoreMean, errors);
        validateMeanStdArray("scoreStd", scoreStd, errors);
        validateMeanStdArray("linearMean", linearMean, errors);
        validateMeanStdArray("linearStd", linearStd, errors);
        validateAngleDict(angleDict, errors);

        // ==================== GPU参数验证 ====================
        if (gpuId < 0) {
            errors.add("gpuId must be >= 0");
        }

        // ==================== 抛出验证异常 ====================
        if (!errors.isEmpty()) {
            throw new IllegalArgumentException("Invalid OCRConfig: " + String.join("; ", errors));
        }
    }

    // ==================== 验证方法 ====================

    /**
     * 验证必需文件是否存在且可读
     */
    private static void validateRequiredFile(String field, String path, List<String> errors) {
        if (isBlank(path)) {
            errors.add(field + " must not be blank");
            return;
        }

        File file = new File(path);

        if (!file.exists()) {
            errors.add(field + " file does not exist: " + path);
            return;
        }

        if (!file.isFile()) {
            errors.add(field + " is not a file: " + path);
            return;
        }

        if (!file.canRead()) {
            errors.add(field + " is not readable: " + path);
        }
    }

    /**
     * 验证调试路径
     */
    private static void validateDebugPath(String debugPath, List<String> errors) {
        if (isBlank(debugPath)) {
            errors.add("debugPath must not be blank when useDebug=true");
            return;
        }

        File dir = new File(debugPath);
        if (dir.exists() && !dir.isDirectory()) {
            errors.add("debugPath is not a directory: " + debugPath);
        } else if (!dir.exists() && !dir.mkdirs()) {
            errors.add("failed to create debugPath directory: " + debugPath);
        }
    }

    /**
     * 验证正整数
     */
    private static void validatePositive(String field, int value, List<String> errors) {
        if (value <= 0) {
            errors.add(field + " must be > 0, current: " + value);
        }
    }

    /**
     * 验证范围在 [0, 1] 之间的浮点数
     */
    private static void validateRange01(String field, float value, List<String> errors) {
        if (!Float.isFinite(value) || value < 0.0f || value > 1.0f) {
            errors.add(field + " must be in [0, 1], current: " + value);
        }
    }

    /**
     * 验证正浮点数
     */
    private static void validateRangePositive(String field, float value, List<String> errors) {
        if (!Float.isFinite(value) || value <= 0.0f) {
            errors.add(field + " must be > 0, current: " + value);
        }
    }

    /**
     * 验证均值和标准差数组
     */
    private static void validateMeanStdArray(String field, float[] array, List<String> errors) {
        if (array == null) {
            errors.add(field + " must not be null");
            return;
        }

        if (array.length != 3) {
            errors.add(field + " must have length " + 3 + ", current: " + array.length);
            return;
        }

        for (int i = 0; i < array.length; i++) {
            if (!Float.isFinite(array[i])) {
                errors.add(field + "[" + i + "] is not a finite number");
            }
        }
    }

    /**
     * 验证角度字典
     */
    private static void validateAngleDict(int[] angleDict, List<String> errors) {
        if (angleDict == null) {
            errors.add("angleDict must not be null");
            return;
        }

        if (angleDict.length == 0) {
            errors.add("angleDict must not be empty");
            return;
        }

        for (int i = 0; i < angleDict.length; i++) {
            int angle = angleDict[i];
            if (angle != 0 && angle != 90 && angle != 180 && angle != 270) {
                errors.add("angleDict[" + i + "] must be one of [0, 90, 180, 270], current: " + angle);
            }
        }
    }

    /**
     * 检查字符串是否为空白
     */
    private static boolean isBlank(String text) {
        return text == null || text.trim().isEmpty();
    }

    // ==================== equals() 和 hashCode() ====================

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        OCRConfig that = (OCRConfig) o;

        // 比较基本类型字段
        return useCls == that.useCls &&
                useDebug == that.useDebug &&
                batchSize == that.batchSize &&
                stride == that.stride &&
                dilateKernelSize == that.dilateKernelSize &&
                clsModelWidth == that.clsModelWidth &&
                clsModelHeight == that.clsModelHeight &&
                isDilation == that.isDilation &&
                Float.compare(that.bitThresh, bitThresh) == 0 &&
                Float.compare(that.unclipRatio, unclipRatio) == 0 &&
                boxLimit == that.boxLimit &&
                boxMinSize == that.boxMinSize &&
                boxMinArea == that.boxMinArea &&
                Float.compare(that.boxMinAspectRatio, boxMinAspectRatio) == 0 &&
                Float.compare(that.boxThresh, boxThresh) == 0 &&
                Float.compare(that.clsThresh, clsThresh) == 0 &&
                Float.compare(that.recThresh, recThresh) == 0 &&
                useGpu == that.useGpu &&
                gpuId == that.gpuId &&
                numThreads == that.numThreads &&
                // 比较字符串字段
                Objects.equals(detModelPath, that.detModelPath) &&
                Objects.equals(clsModelPath, that.clsModelPath) &&
                Objects.equals(recModelPath, that.recModelPath) &&
                Objects.equals(dictPath, that.dictPath) &&
                Objects.equals(debugPath, that.debugPath) &&
                // 比较数组字段
                Arrays.equals(scoreMean, that.scoreMean) &&
                Arrays.equals(scoreStd, that.scoreStd) &&
                Arrays.equals(linearMean, that.linearMean) &&
                Arrays.equals(linearStd, that.linearStd) &&
                Arrays.equals(angleDict, that.angleDict);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(detModelPath, clsModelPath, recModelPath, dictPath, debugPath,
                useCls, useDebug, batchSize, stride, dilateKernelSize,
                clsModelWidth, clsModelHeight, isDilation, bitThresh,
                unclipRatio, boxLimit, boxMinSize, boxMinArea,
                boxMinAspectRatio, boxThresh, clsThresh, recThresh,
                useGpu, gpuId, numThreads);
        result = 31 * result + Arrays.hashCode(scoreMean);
        result = 31 * result + Arrays.hashCode(scoreStd);
        result = 31 * result + Arrays.hashCode(linearMean);
        result = 31 * result + Arrays.hashCode(linearStd);
        result = 31 * result + Arrays.hashCode(angleDict);
        return result;
    }
}