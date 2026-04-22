package com.ocr.paddleocr.process;

import com.ocr.paddleocr.config.OCRConfig;

public class ClsProcessor {

    private final ModelManager modelManager;
    private final OCRConfig config;

    public ClsProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.config = modelManager.getConfig();
    }
}
