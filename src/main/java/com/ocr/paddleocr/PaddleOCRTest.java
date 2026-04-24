package com.ocr.paddleocr;

import com.ocr.paddleocr.service.OCRService;
import lombok.extern.slf4j.Slf4j;

import java.io.File;

@Slf4j
public class PaddleOCRTest {

    public static void main(String[] args) {
        // 测试图片路径 - 请修改为实际的图片路径
//        String testImagePath = "C:\\Users\\26221\\Desktop\\paddle-ocr-onnx\\INE_FRONT4.jpeg";
//        String testImagePath = "src/main/java/resources/test/QQ20260321-104004.png";
//        String testImagePath = "src/main/java/resources/test/20260409-141403.jpg";
        String testImagePath = "src/main/java/resources/test/chi_test.jpg";

        // 运行测试
        runBasicTest(testImagePath);
    }

    public static void runBasicTest(String imagePath) {
        log.info("========== 基础识别测试 ==========");
        log.info("图片路径: {}", imagePath);

        try{
            // 执行OCR识别
            String jsonResult = OCRService.recognize(imagePath);

            // 打印识别结果
            System.out.println(jsonResult);

        } catch (Exception e) {
            log.error("OCR识别失败", e);
        }
    }
}
