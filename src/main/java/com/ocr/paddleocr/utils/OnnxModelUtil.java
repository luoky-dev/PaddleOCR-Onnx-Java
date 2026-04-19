package com.ocr.paddleocr.utils;
import ai.onnxruntime.*;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class OnnxModelUtil {

    private OrtEnvironment env;

    public void OnnxModelInfo() {
        this.env = OrtEnvironment.getEnvironment();
    }

    /**
     * 获取检测模型的输入尺寸
     *
     * @param session ONNX会话
     * @return 输入尺寸 [height, width]
     */
    public int[] getDetInputSize(OrtSession session) {
        try {
            var inputInfo = session.getInputInfo().values().iterator().next();
            if (inputInfo.getInfo() instanceof TensorInfo) {
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();

                // 形状格式: [batch, channels, height, width] 或 [batch, height, width, channels]
                if (shape.length >= 4) {
                    // 判断是CHW还是HWC格式
                    if (shape[1] == 3 || shape[1] == 1) {
                        // CHW格式: [N, C, H, W]
                        return new int[]{(int) shape[2], (int) shape[3]};
                    } else {
                        // HWC格式: [N, H, W, C]
                        return new int[]{(int) shape[1], (int) shape[2]};
                    }
                }
            }
        } catch (Exception e) {
            log.warn("获取检测模型输入尺寸失败，使用默认值: {}", e.getMessage());
        }
        return new int[]{960, 960}; // 默认值
    }

    /**
     * 获取识别模型的输入尺寸
     *
     * @param session ONNX会话
     * @return 输入尺寸 [height, width]
     */
    public int[] getRecInputSize(OrtSession session) {
        try {
            var inputInfo = session.getInputInfo().values().iterator().next();
            if (inputInfo.getInfo() instanceof TensorInfo) {
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();

                if (shape.length >= 4) {
                    // 识别模型通常是 [batch, channels, height, width]
                    return new int[]{(int) shape[2], (int) shape[3]};
                }
            }
        } catch (Exception e) {
            log.warn("获取识别模型输入尺寸失败，使用默认值: {}", e.getMessage());
        }
        return new int[]{48, 320}; // 默认值
    }

    /**
     * 获取分类模型的输入尺寸
     *
     * @param session ONNX会话
     * @return 输入尺寸 [height, width]
     */
    public int[] getClsInputSize(OrtSession session) {
        try {
            var inputInfo = session.getInputInfo().values().iterator().next();
            if (inputInfo.getInfo() instanceof TensorInfo) {
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();

                if (shape.length >= 4) {
                    // 分类模型通常是 [batch, channels, height, width]
                    return new int[]{(int) shape[2], (int) shape[3]};
                }
            }
        } catch (Exception e) {
            log.warn("获取分类模型输入尺寸失败，使用默认值: {}", e.getMessage());
        }
        return new int[]{48, 192}; // 默认值
    }

    /**
     * 获取模型的输入名称
     *
     * @param session ONNX会话
     * @return 输入名称
     */
    public String getInputName(OrtSession session) {
        try {
            return session.getInputInfo().keySet().iterator().next();
        } catch (Exception e) {
            log.warn("获取输入名称失败，使用默认值: x");
            return "x";
        }
    }

    /**
     * 获取模型的输出名称
     *
     * @param session ONNX会话
     * @return 输出名称
     */
    public String getOutputName(OrtSession session) {
        try {
            return session.getOutputInfo().keySet().iterator().next();
        } catch (Exception e) {
            log.warn("获取输出名称失败，使用默认值: output");
            return "output";
        }
    }

    /**
     * 获取识别模型的类别数（字典大小）
     *
     * @param session ONNX会话
     * @return 类别数
     */
    public int getNumClasses(OrtSession session) {
        try {
            var outputInfo = session.getOutputInfo().values().iterator().next();
            if (outputInfo.getInfo() instanceof TensorInfo) {
                TensorInfo tensorInfo = (TensorInfo) outputInfo.getInfo();
                long[] shape = tensorInfo.getShape();

                if (shape.length >= 2) {
                    // 输出形状: [seq_len, num_classes] 或 [batch, seq_len, num_classes]
                    return (int) shape[shape.length - 1];
                }
            }
        } catch (Exception e) {
            log.warn("获取类别数失败: {}", e.getMessage());
        }
        return -1;
    }

    /**
     * 判断模型输入是否为CHW格式
     *
     * @param session ONNX会话
     * @return true: CHW (N,C,H,W), false: HWC (N,H,W,C)
     */
    public boolean isCHWFormat(OrtSession session) {
        try {
            var inputInfo = session.getInputInfo().values().iterator().next();
            if (inputInfo.getInfo() instanceof TensorInfo) {
                TensorInfo tensorInfo = (TensorInfo) inputInfo.getInfo();
                long[] shape = tensorInfo.getShape();

                if (shape.length >= 4) {
                    // 如果第二维是3或1，通常是CHW格式
                    return shape[1] == 3 || shape[1] == 1;
                }
            }
        } catch (Exception e) {
            log.warn("判断格式失败，默认使用CHW格式");
        }
        return true; // 默认CHW格式
    }

    /**
     * 打印模型信息（调试用）
     *
     * @param session ONNX会话
     * @param modelName 模型名称
     */
    public void printModelInfo(OrtSession session, String modelName) {
        log.info("========== {} 模型信息 ==========", modelName);

        try {
            // 输入信息
            log.info("输入:");
            for (var entry : session.getInputInfo().entrySet()) {
                String name = entry.getKey();
                if (entry.getValue().getInfo() instanceof TensorInfo) {
                    TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
                    log.info("  - 名称: {}, 形状: {}", name, Arrays.toString(tensorInfo.getShape()));
                }
            }

            // 输出信息
            log.info("输出:");
            for (var entry : session.getOutputInfo().entrySet()) {
                String name = entry.getKey();
                if (entry.getValue().getInfo() instanceof TensorInfo) {
                    TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
                    log.info("  - 名称: {}, 形状: {}", name, Arrays.toString(tensorInfo.getShape()));
                }
            }
        } catch (Exception e) {
            log.error("获取模型信息失败: {}", e.getMessage());
        }

        log.info("=================================");
    }

    /**
     * 关闭环境
     */
    public void close() {
        if (env != null) {
            env.close();
        }
    }
}
