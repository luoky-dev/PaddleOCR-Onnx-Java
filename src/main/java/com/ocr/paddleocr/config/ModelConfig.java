package com.ocr.paddleocr.config;

import lombok.Getter;

@Getter
public class ModelConfig {
    // 检测模型最大边长（paddleOCR官方默认 960）
    private final int detMaxSide = 960;
    // 对齐倍数（paddleOCR官方要求 32 的倍数）
    private final int detStride = 32;
    // 减均值除方差 (Z-score Normalization)
    // ImageNet RGB均值
    private final float[] scoreMean = {0.485f, 0.456f, 0.406f};
    // ImageNet RGB标准差
    private final float[] scoreStd = {0.229f, 0.224f, 0.225f};
    // 线性 (Linear Scaling)
    // 线性到 [-1,1] RGB均值
    private final float[] linearMean = {0.5f, 0.5f, 0.5f};
    // 线性到 [-1,1] RGB标准差
    private final float[] linearStd = {0.5f, 0.5f, 0.5f};
    // 膨胀核大小（paddleOCR官方默认 3）
    private final int dilateKernelSize = 3;
    // 最大候选框数量
    private final int maxCandidates = 1000;
    // 官方 epsilon 值
    private final float epsilon = 0.002f;
    // 方向分类模型输入宽度
    private final int clsModelWith = 320;
    // 方向分类模型输入高度
    private final int clsModelHeight = 48;
    // 方向四分类（支持垂直文本）
    private final int[] fallbackAngleMap = {0, 90, 180, 270};
    // 识别模型输入宽度
    private final int recModelWith = 320;
    // 识别模型输入高度
    private final int recModelHeight = 48;
    // 字典空格下标
    private final int blankIndex = 0;
}
