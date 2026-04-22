package com.ocr.paddleocr.process;

import com.ocr.paddleocr.config.OCRConfig;

public class DetProcessor {

    private final ModelManager modelManager;
    private final OCRConfig config;

    public DetProcessor(ModelManager modelManager) {
        this.modelManager = modelManager;
        this.config = modelManager.getConfig();
    }
}
