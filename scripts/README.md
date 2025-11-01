# OpenPose 视频处理工具

这个脚本使用OpenCV的DNN模块来实现人体姿态识别，可以处理攀岩视频并输出带有骨架标记的视频。

## 环境要求

- Python 3.8+
- OpenCV
- NumPy
- Requests (用于下载模型)

## 使用方法

1. **准备环境**

   已经创建了名为`openpose`的conda环境，并安装了必要的依赖：
   ```bash
   conda activate openpose
   ```

2. **处理视频**

   在`scripts`目录下运行：
   ```bash
   python process_video.py --video_path path_to_your_climbing_video.mp4 --output_path output_with_pose.mp4
   ```

   参数说明：
   - `--video_path`: 输入视频文件路径（默认为input.mp4）
   - `--output_path`: 输出视频文件路径（默认为output.mp4）

## 工作原理

脚本使用两种方式进行姿态检测：

1. **主要方法**：使用OpenCV的DNN模块加载预训练的人体姿态估计模型（基于COCO数据集）
   - 自动下载所需的模型文件
   - 检测18个关键点并绘制骨架

2. **备选方法**：如果模型加载失败，使用OpenCV的级联分类器进行简单的人脸检测和身体轮廓绘制

## 注意事项

- 处理高清视频可能会比较慢，建议先使用较低分辨率的视频进行测试
- 第一次运行时会下载模型文件，可能需要一些时间
- 确保有足够的磁盘空间存储输出视频

## 示例

处理攀岩视频的命令示例：
```bash
python process_video.py --video_path climbing_video.mp4 --output_path climbing_with_pose.mp4
```

输出视频将显示攀岩者的骨架，包括头部、躯干、手臂和腿部的关键点和连接线。