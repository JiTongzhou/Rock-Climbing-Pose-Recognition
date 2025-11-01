import cv2
import os
import sys
import numpy as np
import argparse
from collections import deque

def get_mediapipe_pose():
    # 尝试导入mediapipe库
    try:
        import mediapipe as mp
        print("成功导入MediaPipe库")
        
        # 初始化MediaPipe姿态检测 - 调整参数以提高稳定性
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,  # 视频模式
            model_complexity=2,       # 使用最复杂的模型以获得最佳效果
            smooth_landmarks=True,    # 启用内置平滑
            min_detection_confidence=0.6,  # 提高检测阈值以减少误检
            min_tracking_confidence=0.7,   # 提高跟踪阈值以增强稳定性
            smooth_segmentation=True,      # 启用分割平滑
            enable_segmentation=False      # 禁用分割以提高性能
        )
        
        # 获取绘图工具
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化历史缓存
        init_landmark_history()
        
        return pose, mp_pose, mp_drawing, mp_drawing_styles
    except ImportError:
        print("未找到MediaPipe库，请安装")
        return None

# 用于存储历史关键点的缓存
landmark_history = None
HISTORY_SIZE = 5  # 历史帧数缓存

# 初始化关键点历史缓存
def init_landmark_history():
    global landmark_history
    # 为每个关键点创建一个双端队列，用于存储历史坐标
    landmark_history = [deque(maxlen=HISTORY_SIZE) for _ in range(33)]  # MediaPipe有33个关键点

# 应用移动平均滤波来平滑关键点
def apply_smoothing(landmark_idx, x, y, visibility):
    if landmark_history is None:
        init_landmark_history()
    
    # 根据置信度决定权重
    weight = visibility if visibility > 0.5 else 0.3
    
    # 添加当前点到历史记录
    landmark_history[landmark_idx].append((x, y, weight))
    
    # 如果历史记录不足，直接返回当前点
    if len(landmark_history[landmark_idx]) < 2:
        return x, y
    
    # 计算加权移动平均
    total_weight = sum(pt[2] for pt in landmark_history[landmark_idx])
    if total_weight == 0:
        return x, y
    
    avg_x = sum(pt[0] * pt[2] for pt in landmark_history[landmark_idx]) / total_weight
    avg_y = sum(pt[1] * pt[2] for pt in landmark_history[landmark_idx]) / total_weight
    
    return avg_x, avg_y

def detect_pose_with_mediapipe(frame, pose, mp_pose, mp_drawing, mp_drawing_styles):
    # 初始化历史缓存（如果尚未初始化）
    if landmark_history is None:
        init_landmark_history()
    
    # 转换图像为RGB格式（MediaPipe需要RGB输入）
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    results = pose.process(image_rgb)
    
    # 绘制姿态关键点和连接线
    if results.pose_landmarks:
        h, w, _ = frame.shape
        smoothed_landmarks = []
        
        # 首先计算所有平滑后的关键点
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # 获取原始坐标
            raw_x = landmark.x * w
            raw_y = landmark.y * h
            visibility = landmark.visibility
            
            # 应用平滑处理
            smoothed_x, smoothed_y = apply_smoothing(idx, raw_x, raw_y, visibility)
            smoothed_landmarks.append((int(smoothed_x), int(smoothed_y), visibility))
        
        # 绘制骨架连接线
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # 获取平滑后的关键点坐标
            start_x, start_y, start_vis = smoothed_landmarks[start_idx]
            end_x, end_y, end_vis = smoothed_landmarks[end_idx]
            
            # 只有当两个点的可见度都足够高时才绘制连接线
            if start_vis > 0.3 and end_vis > 0.3:
                # 根据不同的身体部位使用不同的颜色，类似OpenPose
                # 头颈部 - 紫色
                if (start_idx, end_idx) in [(0, 1), (1, 2), (2, 3), (3, 4)] or \
                   (end_idx, start_idx) in [(0, 1), (1, 2), (2, 3), (3, 4)]:
                    color = (255, 0, 255)  # 紫色
                # 躯干 - 红色
                elif (start_idx, end_idx) in [(1, 5), (5, 6), (6, 7), (7, 8), (1, 11), (11, 12), (12, 13), (13, 14)] or \
                     (end_idx, start_idx) in [(1, 5), (5, 6), (6, 7), (7, 8), (1, 11), (11, 12), (12, 13), (13, 14)]:
                    color = (0, 0, 255)  # 红色
                # 左臂 - 绿色
                elif (start_idx, end_idx) in [(11, 15), (15, 17)] or \
                     (end_idx, start_idx) in [(11, 15), (15, 17)]:
                    color = (0, 255, 0)  # 绿色
                # 右臂 - 橙色
                elif (start_idx, end_idx) in [(12, 16), (16, 18)] or \
                     (end_idx, start_idx) in [(12, 16), (16, 18)]:
                    color = (0, 165, 255)  # 橙色
                # 左腿 - 蓝色
                elif (start_idx, end_idx) in [(23, 25), (25, 27)] or \
                     (end_idx, start_idx) in [(23, 25), (25, 27)]:
                    color = (255, 0, 0)  # 蓝色
                # 右腿 - 青色
                elif (start_idx, end_idx) in [(24, 26), (26, 28)] or \
                     (end_idx, start_idx) in [(24, 26), (26, 28)]:
                    color = (255, 255, 0)  # 青色
                else:
                    color = (0, 255, 255)  # 黄色
                
                # 绘制连接线，线条粗细根据置信度调整
                thickness = int(2 + (min(start_vis, end_vis) * 1.5))
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
        
        # 绘制关键点（小圆点）
        for idx, (x, y, visibility) in enumerate(smoothed_landmarks):
            # 根据置信度调整点的大小
            if visibility > 0.7:
                # 高置信度：大圆点
                cv2.circle(frame, (x, y), 6, (0, 0, 0), -1)  # 黑色外圈
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)  # 白色中心点
            elif visibility > 0.5:
                # 中等置信度：中等圆点
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)  # 黑色外圈
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)  # 白色中心点
            elif visibility > 0.3:
                # 低置信度：小圆点
                cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)  # 黑色外圈
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # 白色中心点
    
    return frame

def get_openpose_model_from_caffe():
    # 尝试使用OpenCV的DNN模块加载OpenPose模型
    try:
        # 定义模型文件路径
        proto_file = "pose_deploy_linevec_faster_4_stages.prototxt"
        weights_file = "pose_iter_160000.caffemodel"
        
        # 尝试加载模型
        net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception as e:
        print(f"加载Caffe模型失败: {e}")
        return None

def detect_pose_with_caffe(frame, net):
    # 人体关键点连接对
    POSE_PAIRS = [
        [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
        [0, 14], [0, 15], [14, 16], [15, 17]
    ]
    
    # 模型输入参数
    inWidth = 368
    inHeight = 368
    inScaleFactor = 1.0 / 255
    mean = (0, 0, 0)
    
    # 创建blob
    blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), mean, swapRB=False, crop=False)
    
    # 设置输入并前向传播
    net.setInput(blob)
    output = net.forward()
    
    # 获取图像尺寸
    H, W = frame.shape[:2]
    
    # 关键点列表
    points = []
    
    # 获取关键点
    for i in range(18):  # COCO数据集有18个关键点
        # 置信度图
        probMap = output[0, i, :, :]
        
        # 找到最大置信度位置
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # 将坐标调整到原始图像尺寸
        x = (W * point[0]) / output.shape[3]
        y = (H * point[1]) / output.shape[2]
        
        # 如果置信度足够高，则添加关键点
        if prob > 0.1:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    # 绘制骨架，使用类似OpenPose的颜色编码
    # 颜色映射，为不同的身体部位分配不同的颜色
    colors = [
        (255, 0, 255),  # 头部 - 紫色
        (0, 0, 255),    # 颈部/躯干 - 红色
        (0, 255, 0),    # 右臂 - 绿色
        (0, 255, 0),    # 右前臂 - 绿色
        (0, 255, 0),    # 右手 - 绿色
        (0, 165, 255),  # 左臂 - 橙色
        (0, 165, 255),  # 左前臂 - 橙色
        (0, 165, 255),  # 左手 - 橙色
        (255, 0, 0),    # 右大腿 - 蓝色
        (255, 0, 0),    # 右小腿 - 蓝色
        (255, 0, 0),    # 右脚 - 蓝色
        (255, 255, 0),  # 左大腿 - 青色
        (255, 255, 0),  # 左小腿 - 青色
        (255, 255, 0),  # 左脚 - 青色
        (128, 0, 128),  # 右眼 - 紫色
        (128, 0, 128),  # 左眼 - 紫色
        (128, 0, 128),  # 右耳 - 紫色
        (128, 0, 128)   # 左耳 - 紫色
    ]
    
    # 绘制骨架连接线
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        
        if points[partA] and points[partB]:
            # 使用partA的颜色来绘制连接线
            color = colors[partA] if partA < len(colors) else (0, 255, 255)
            # 绘制连接线
            cv2.line(frame, points[partA], points[partB], color, 2)
    
    # 绘制关键点
    for i, point in enumerate(points):
        if point:
            color = colors[i] if i < len(colors) else (0, 255, 255)
            cv2.circle(frame, point, 8, color, thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, point, 4, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
    
    return frame

def get_pose_model():
    # 首先尝试使用MediaPipe
    mediapipe_result = get_mediapipe_pose()
    if mediapipe_result:
        return mediapipe_result, "mediapipe"
    
    # 如果MediaPipe不可用，尝试使用Caffe模型
    net = get_openpose_model_from_caffe()
    if net:
        return net, "caffe"
    
    # 如果都不可用，返回None
    return None, None

def detect_pose(frame, pose_model):
    # 根据模型类型调用不同的检测函数
    if isinstance(pose_model, tuple) and pose_model[1] == "mediapipe":
        pose, mp_pose, mp_drawing, mp_drawing_styles = pose_model[0]
        return detect_pose_with_mediapipe(frame, pose, mp_pose, mp_drawing, mp_drawing_styles)
    elif pose_model[1] == "caffe":
        return detect_pose_with_caffe(frame, pose_model[0])
    else:
        # 如果没有有效的模型，返回原始帧
        return frame

def process_video(video_path, output_path):
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在")
        return False
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'")
        return False
    
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息：{width}x{height}, {fps}fps, 共{total_frames}帧")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 加载姿态估计模型
    print("正在加载姿态估计模型...")
    model_info, model_type = get_pose_model()
    
    if model_type == "mediapipe":
        print("使用MediaPipe模型进行姿态检测")
        pose, mp_pose, mp_drawing, mp_drawing_styles = model_info
    elif model_type == "caffe":
        print("使用OpenCV DNN模型进行姿态检测")
        net = model_info
    else:
        print("警告：无法加载姿态检测模型")
        print("启用备选方案：简化的姿态检测")
    
    # 处理视频帧
    frame_count = 0
    # 重置关键点历史缓存
    init_landmark_history()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根据模型类型处理当前帧
            if model_type == "mediapipe":
                # 创建帧的副本以避免修改原始帧
                frame_copy = frame.copy()
                processed_frame = detect_pose_with_mediapipe(frame_copy, pose, mp_pose, mp_drawing, mp_drawing_styles)
            elif model_type == "caffe":
                processed_frame = detect_pose_with_caffe(frame, net)
            else:
                # 备选方案：使用简化的姿态检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # 绘制人脸框
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # 简单的身体轮廓绘制（基于人脸位置估计）
                    body_top = y + h
                    body_bottom = min(y + h * 4, height)
                    body_center = x + w // 2
                    body_width = w * 2
                    
                    # 绘制身体轮廓
                    cv2.rectangle(frame, 
                                 (body_center - body_width // 2, body_top), 
                                 (body_center + body_width // 2, body_bottom), 
                                 (0, 255, 0), 2)
                    
                    # 绘制简单的手臂轮廓
                    arm_length = body_width // 2
                    cv2.line(frame, 
                            (body_center - body_width // 2, body_top + h // 2), 
                            (body_center - body_width // 2 - arm_length, body_top + h), 
                            (0, 0, 255), 2)
                    cv2.line(frame, 
                            (body_center + body_width // 2, body_top + h // 2), 
                            (body_center + body_width // 2 + arm_length, body_top + h), 
                            (0, 0, 255), 2)
                
                processed_frame = frame
            
            # 写入输出视频
            out.write(processed_frame)
            
            # 显示进度
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"处理进度: {frame_count}/{total_frames} 帧 ({progress:.1f}%)", end='\r')
    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        # 释放资源
        if model_type == "mediapipe" and model_info:
            pose.close()
        cap.release()
        out.release()
        print("\n视频处理完成！")
        print(f"输出文件保存至: {output_path}")
    
    return True

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="input.mp4", help="Process a video. Read all standard formats (mp4, avi, etc.).")
    parser.add_argument("--output_path", default="output.mp4", help="Path to save the output video.")
    args = parser.parse_args()
    
    print(f"输入视频: {args.video_path}")
    print(f"输出视频: {args.output_path}")
    
    # 提示用户安装mediapipe以获得更好的姿态检测效果
    try:
        import mediapipe
        print("Mediapipe库已安装，将使用更好的姿态检测效果")
    except ImportError:
        print("\n警告：未检测到Mediapipe库")
        print("推荐安装Mediapipe库以获得更好的姿态检测效果")
        print("安装命令: pip install mediapipe")
    
    # 处理视频
    success = process_video(args.video_path, args.output_path)
    
    if success:
        print("\n姿态检测视频处理成功！")
    else:
        print("\n姿态检测视频处理失败！")
        sys.exit(1)