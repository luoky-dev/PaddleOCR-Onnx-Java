# PaddleOCR-Onnx-Java
基于 ONNX Runtime 的 PaddleOCR Java 实现，支持文本检测、识别和方向分类。
> 总结：**一个开箱即用的 Java OCR SDK，基于 ONNX Runtime，在 CPU/GPU 上都能跑通“检测 + 方向分类 + 识别”全流程。**

PaddleOCR-Onnx-Java 是一个面向 Java 业务系统的 OCR 推理组件。 
基于 ONNX Runtime + OpenCV 的 Java OCR SDK，提供从图片路径到 OCR 结果 JSON 的端到端能力，覆盖文本检测（det）、方向分类（cls）和文本识别（rec）。 
输入图片路径，输出结构化识别结果（文本、置信度、位置、耗时），适合快速接入后端服务或离线任务。

## 功能矩阵

| 能力 | 是否支持 | 说明 |
|---|---|---|
| 文本检测（Det） | ✅ | 支持倾斜文本区域检测 |
| 方向分类（Cls） | ✅ | 可对 0/90/180/270 方向做纠正 |
| 文本识别（Rec） | ✅ | CTC 解码，输出文本与置信度 |
| 多语言模型切换 | ✅ | 通过模型路径与字典配置切换 |
| 批量推理 | ✅ | 可配置 `batchSize` |
| 多线程推理 | ✅ | 可配置 `numThreads` |
| GPU 推理 | ✅ | `useGpu=true` 优先 CUDA，失败自动回退 CPU |
| Debug 可视化导出 | ✅ | 检测框/热力图/叠加图等中间结果导出 |

## 快速开始

### 1. 引入依赖

默认 CPU 依赖：

```xml
<dependency>
  <groupId>com.ocr</groupId>
  <artifactId>paddleocr-onnx</artifactId>
  <version>1.0.0</version>
</dependency>
```

> 构建本项目时，默认使用 `onnxruntime`（CPU）；使用 `-Pgpu` 可切换 `onnxruntime_gpu`。

### 2. 准备模型与字典

在 `OCRConfig` 中配置以下路径：
- `detModelPath`
- `clsModelPath`（可选，关闭 `useCls` 可不配）
- `recModelPath`
- `dictPath`

### 3. 代码示例

```java
import com.ocr.paddleocr.config.OCRConfig;
import com.ocr.paddleocr.service.OCRService;

public class Demo {
    public static void main(String[] args) {
        OCRConfig config = OCRConfig.builder()
                .detModelPath("models/chi/det_model.onnx")
                .clsModelPath("models/chi/cls_model.onnx")
                .recModelPath("models/chi/rec_model.onnx")
                .dictPath("models/chi/ppocr_keys_v1.txt")
                .useCls(true)
                .batchSize(16)
                .numThreads(4)
                .useGpu(false) // true: 优先CUDA，失败自动回退CPU
                .gpuId(0)
                .build();

        String resultJson = OCRService.recognize(config, "test/chi_test.jpg");
        System.out.println(resultJson);
    }
}
```

### 4. GPU 构建（可选）

```bash
mvn -Pgpu clean package
```

### 5. 常见输出

返回 JSON 典型字段：
- `success`: 是否成功
- `processingTime`: 总耗时（ms）
- `words`: 识别结果列表（`text`、`confidence`、`box`）

## 适用场景

- 证件/票据/截图文字提取
- 表单录入自动化
- 图片内容检索与结构化入库

## License

本项目采用仓库中的 `LICENSE` 协议发布。
