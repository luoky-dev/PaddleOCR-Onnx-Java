package com.ocr.paddleocr.config;

import lombok.Data;
import lombok.Getter;

@Getter
public class ModelConfig {
    // 检测模型最大边长（官方默认 960）
    private final int detMaxSideLen = 960;
    // 对齐倍数（官方要求 32 的倍数）
    private final int resizeAlign = 32;
    // RGB 通道的均值和标准差（ImageNet 标准化参数）
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std = {0.229f, 0.224f, 0.225f};
    // 膨胀核大小
    private final int dilateKernelSize = 3;
    // 最大候选框数量
    private final int maxCandidates = 1000;
    // 官方 epsilon 值
    private final float epsilon = 0.002f;
    // 方向分类模型输入宽度
    private final int clsModelWith = 192;
    // 方向分类模型输入高度
    private final int clsModelHeight = 48;
    // 方向四分类（支持垂直文本）
    private final int[] fallbackAngleMap = {0, 90, 180, 270};
}
