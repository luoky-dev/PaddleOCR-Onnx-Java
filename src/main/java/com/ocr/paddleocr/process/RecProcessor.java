package com.ocr.paddleocr.process;

import com.ocr.paddleocr.config.OCRConfig;

public class RecProcessor {

    private final ModelManager modelManager;
    private final OCRConfig config;

    public RecProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.config = modelManager.getConfig();
    }
}
