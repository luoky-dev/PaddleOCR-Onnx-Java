package com.ocr.paddleocr.process;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.ocr.paddleocr.config.ModelConfig;
import com.ocr.paddleocr.config.OCRConfig;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Data
@NoArgsConstructor
public class ModelManager implements AutoCloseable {
    private static volatile ModelManager instance;
    private OrtEnvironment env;
    private OrtSession detSession;
    private OrtSession recSession;
    private OrtSession clsSession;
    private OCRConfig ocrConfig;
    private ModelConfig modelConfig;
    /**
     * 初始化标志
     */
    private volatile boolean initialized = false;
    /**
     * OpenCV加载标志
     */
    private static final AtomicBoolean openCVLoaded = new AtomicBoolean(false);

    /**
     * 获取单例实例
     */
    public static ModelManager getInstance() {
        if (instance == null) {
            synchronized (ModelManager.class) {
                if (instance == null) {
                    instance = new ModelManager();
                }
            }
        }
        return instance;
    }

    /**
     * 初始化模型管理器
     */
    public synchronized void init(OCRConfig ocrConfig) throws Exception {
        if (initialized) {
            log.warn("模型管理器已初始化, 跳过重复加载");
            return;
        }

        long startTime = System.currentTimeMillis();
        log.info("开始初始化模型管理器");

        this.ocrConfig = ocrConfig;
        this.modelConfig = new ModelConfig();

        // 加载OpenCV
        loadOpenCV();

        // 加载ONNX模型
        loadONNXModels();

        long loadTimeMs = System.currentTimeMillis() - startTime;
        this.initialized = true;

        log.info("模型管理器初始化完成, 耗时: {} ms", loadTimeMs);
    }

    /**
     * 加载OpenCV库
     */
    private void loadOpenCV() {
        if (openCVLoaded.get()) {
            return;
        }
        try {
            nu.pattern.OpenCV.loadLocally();
            openCVLoaded.set(true);
            log.info("OpenCV加载成功");
        } catch (Exception e) {
            log.error("OpenCV加载失败", e);
            throw new RuntimeException("OpenCV加载失败", e);
        }
    }

    /**
     * 加载ONNX模型
     */
    private void loadONNXModels() throws OrtException {
        log.info("开始加载ONNX模型");

        env = OrtEnvironment.getEnvironment();
        SessionOptions sessionOptions = buildSessionOptions();
        try {
            if (ocrConfig.getDetModelPath() != null) {
                log.info("加载检测模型: {}", ocrConfig.getDetModelPath());
                detSession = env.createSession(ocrConfig.getDetModelPath(), sessionOptions);
                log.info("检测模型加载完成");
            }

            if (ocrConfig.getRecModelPath() != null) {
                log.info("加载识别模型: {}", ocrConfig.getRecModelPath());
                recSession = env.createSession(ocrConfig.getRecModelPath(), sessionOptions);
                log.info("识别模型加载完成");
            }

            if (ocrConfig.isUseCls() && ocrConfig.getClsModelPath() != null) {
                log.info("加载分类模型: {}", ocrConfig.getClsModelPath());
                clsSession = env.createSession(ocrConfig.getClsModelPath(), sessionOptions);
                log.info("分类模型加载完成");
            }
        } finally {
            closeSessionOptionsQuietly(sessionOptions);
        }
    }

    /**
     * 构建ONNX Runtime会话配置选项
     * 根据配置决定使用CPU还是GPU，支持GPU失败时自动降级到CPU
     *
     * @return 配置好的SessionOptions对象
     * @throws OrtException ONNX Runtime异常
     */
    private SessionOptions buildSessionOptions() throws OrtException {

        // ========== CPU模式 ==========
        // 如果配置不使用GPU，直接返回CPU配置
        if (!ocrConfig.isUseGpu()) {
            SessionOptions cpuOptions = new SessionOptions();
            applyCommonSessionOptions(cpuOptions);
            log.info("使用 CPU 执行配置, 线程数: {}", ocrConfig.getNumThreads());
            return cpuOptions;
        }

        // ========== GPU模式 ==========
        int gpuId = ocrConfig.getGpuId();
        SessionOptions gpuOptions = new SessionOptions();

        try {
            // 应用通用配置
            applyCommonSessionOptions(gpuOptions);
            // 尝试启用CUDA提供程序
            enableCudaProvider(gpuOptions, gpuId);

            log.info("使用 GPU 执行配置(CUDA), gpuId: {}, 线程数: {}",
                    gpuId, ocrConfig.getNumThreads());
            return gpuOptions;

        } catch (Exception e) {
            // GPU初始化失败，释放资源并降级到CPU
            closeSessionOptionsQuietly(gpuOptions);
            log.warn("启用CUDA提供程序失败, 回退到CPU配置, gpuId: {}", gpuId, e);

            // 创建CPU降级配置
            SessionOptions cpuFallback = new SessionOptions();
            applyCommonSessionOptions(cpuFallback);
            log.info("回退到 CPU 执行配置, 线程数: {}", ocrConfig.getNumThreads());
            return cpuFallback;
        }
    }

    /**
     * 应用通用的Session配置
     * 这些配置对CPU和GPU模式都适用
     *
     * @param sessionOptions 要配置的SessionOptions对象
     * @throws OrtException ONNX Runtime异常
     */
    private void applyCommonSessionOptions(SessionOptions sessionOptions)
            throws OrtException {

        // 设置优化级别：全部优化（最高性能）
        // ALL_OPT 启用量化、算子融合、内存优化等
        sessionOptions.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);

        // 设置跨算子并行线程数
        // 控制不同算子之间的并行执行
        sessionOptions.setInterOpNumThreads(ocrConfig.getNumThreads());

        // 设置算子内部并行线程数
        // 控制单个算子内部的并行执行（如矩阵乘法）
        sessionOptions.setIntraOpNumThreads(ocrConfig.getNumThreads());
    }

    /**
     * 启用CUDA提供程序（GPU加速）
     * 兼容不同版本的ONNX Runtime Java API
     * - 新版本：addCUDA(int deviceId)
     * - 旧版本：addCUDA() 无参数，默认设备0
     *
     * @param sessionOptions 会话选项
     * @param gpuId GPU设备ID
     * @throws Exception 启用失败时抛出异常
     */
    private void enableCudaProvider(SessionOptions sessionOptions, int gpuId)
            throws Exception {

        // 尝试调用 addCUDA(int) 方法（新版本API）
        if (tryInvokeMethod(sessionOptions, "addCUDA",
                new Class<?>[]{int.class}, new Object[]{gpuId})) {
            return;  // 成功，直接返回
        }

        // 尝试调用 addCUDA() 无参方法（旧版本API）
        if (tryInvokeMethod(sessionOptions, "addCUDA",
                new Class<?>[0], new Object[0])) {
            // 旧版本不支持指定GPU ID，发出警告
            if (gpuId != 0) {
                log.warn("Current ONNX Runtime Java API does not expose addCUDA(int). " +
                        "gpuId={} may be ignored.", gpuId);
            }
            return;
        }

        // 两个方法都不存在，说明当前版本不支持CUDA
        throw new IllegalStateException(
                "Current ONNX Runtime Java API does not support CUDA provider.");
    }

    /**
     * 通过反射尝试调用对象的方法
     * 用于兼容不同版本的API，避免编译时依赖不存在的方法
     *
     * @param target 目标对象
     * @param methodName 方法名
     * @param parameterTypes 参数类型数组
     * @param args 参数值数组
     * @return true表示成功调用，false表示方法不存在
     * @throws Exception 调用失败时抛出异常
     */
    private boolean tryInvokeMethod(Object target, String methodName,
                                    Class<?>[] parameterTypes, Object[] args)
            throws Exception {

        // 获取方法
        Method method;
        try {
            method = target.getClass().getMethod(methodName, parameterTypes);
        } catch (NoSuchMethodException e) {
            // 方法不存在，返回false，让调用方尝试其他方法
            return false;
        }

        // 2. 调用方法
        try {
            method.invoke(target, args);
            return true;
        } catch (InvocationTargetException e) {
            // 提取真正的异常原因
            Throwable cause = e.getCause();
            if (cause instanceof Exception) {
                throw (Exception) cause;
            }
            throw new RuntimeException(cause);
        }
    }

    private void closeSessionOptionsQuietly(SessionOptions sessionOptions) {
        if (sessionOptions == null) {
            return;
        }
        sessionOptions.close();
    }

    public OrtSession getDetSession() {
        checkInitialized();
        return detSession;
    }

    /**
     * 获取识别模型会话
     */
    public OrtSession getRecSession() {
        checkInitialized();
        return recSession;
    }

    /**
     * 获取分类模型会话
     */
    public OrtSession getClsSession() {
        checkInitialized();
        return clsSession;
    }

    /**
     * 获取ONNX环境
     */
    public OrtEnvironment getEnv() {
        checkInitialized();
        return env;
    }

    private void checkInitialized() {
        if (!initialized) {
            throw new IllegalStateException("模型管理器未初始化, 请先调用 init() 方法");
        }
    }

    /**
     * 释放资源
     */
    @Override
    public void close() {
        log.info("开始释放模型资源");

        try {
            if (detSession != null) {
                detSession.close();
                log.debug("检测模型已释放");
            }
            if (recSession != null) {
                recSession.close();
                log.debug("识别模型已释放");
            }
            if (clsSession != null) {
                clsSession.close();
                log.debug("分类模型已释放");
            }
            if (env != null) {
                env.close();
                log.debug("ONNX环境已释放");
            }
        } catch (OrtException e) {
            log.error("释放模型资源失败", e);
        } finally {
            detSession = null;
            recSession = null;
            clsSession = null;
            env = null;
            initialized = false;
        }

        log.info("模型资源释放完成");
    }
}
