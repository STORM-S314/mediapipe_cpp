# MediaPipe组成

Solutions一些高层次的、预制的，可以快速简单地让应用程序获得机器学习的插件

Frameworks是一些低层次的组件，可以针对设备执行更高效机器学习的组件

# 安装教程

https://developers.google.com/mediapipe/framework/getting_started/install

MacOS安装MediaPipe框架依赖

1. Xcode命令行工具
2. Bazel编译工具，建议用bazel的sh脚本安装
3. MediaPipe源码
4. OpenCV
5. FFmpeg
6. Python3
7. Python SIX模块

# Bazel学习

## 构建C++项目

https://bazel.google.cn/start/cpp#understand_the_build_file

# Bazel 规则说明

## Bazel通常规则

https://bazel.google.cn/reference/be/common-definitions?hl=zh-cn#typical.data

## Bazel C++规则

https://bazel.google.cn/reference/be/c-cpp?hl=zh-cn#cc_library

https://zhuanlan.zhihu.com/p/421489117

# MediaPipe手部追踪编译过程



hand_tracking_cpu

依赖tensorflow轻量模型文件：

1. mediapipe/modules/hand_landmark:hand_landmark_full.tflite
2. mediapipe/modules/palm_detection:palm_detection_full.tflite

依赖目标：

1. mediapipe/examples/desktop:demo_run_graph_main
2. mediapipe/graphs/hand_tracking:desktop_tflite_calculators



# Unity与C++混合编程