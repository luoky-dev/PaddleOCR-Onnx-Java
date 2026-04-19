package com.ocr.paddleocr.utils;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.ocr.paddleocr.config.OCRConfig;

public class ModelUtil {

    public static SessionOptions getSessionOptions(OCRConfig config) throws OrtException {
        SessionOptions sessionOptions = new OrtSession.SessionOptions();
        // GPU配置
        if (config.isUseGpu()) {
            sessionOptions.addCUDA(config.getGpuId());
        }
        // CPU线程数配置
        sessionOptions.setIntraOpNumThreads(config.getNumThreads());
        // 设置执行模式
        sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
        return sessionOptions;
    }


}
