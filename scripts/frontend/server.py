from flask import Flask, request, jsonify, send_file, render_template
import os
import tempfile
import subprocess
import uuid
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='.', template_folder='.')

# 确保上传目录存在
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB 限制

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 找到process_video.py脚本的路径
PROCESS_VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'process_video.py')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    # 检查是否有文件上传
    if 'video' not in request.files:
        return jsonify({'error': '没有找到视频文件'}), 400
    
    file = request.files['video']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型是否允许
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    # 生成唯一文件名
    unique_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    # 保存上传的文件
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{unique_id}_input.{file_extension}')
    file.save(input_path)
    
    # 生成输出文件路径
    output_filename = f'{unique_id}_output.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        # 调用process_video.py脚本进行处理
        result = subprocess.run(
            ['conda', 'run', '-n', 'openpose', 'python', PROCESS_VIDEO_PATH, 
             '--video_path', input_path, '--output_path', output_path],
            capture_output=True,
            text=True,
            timeout=3600  # 设置1小时超时
        )
        
        # 检查处理是否成功
        if result.returncode != 0:
            raise Exception(f'视频处理失败: {result.stderr}')
        
        # 检查输出文件是否存在
        if not os.path.exists(output_path):
            raise Exception('输出文件未生成')
        
        # 返回成功响应
        return jsonify({
            'success': True,
            'output_filename': output_filename
        })
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': '视频处理超时'}), 500
    except Exception as e:
        print(f'处理错误: {str(e)}')
        return jsonify({'error': str(e)}), 500
    finally:
        # 清理临时文件
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass

@app.route('/download')
def download_file():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': '缺少文件名参数'}), 400
    
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 404
    
    return send_file(filepath, as_attachment=True, download_name=f'pose_result.mp4')

# 清理过期的输出文件
def cleanup_old_files():
    import time
    current_time = time.time()
    
    # 清理超过24小时的文件
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > 24 * 3600:  # 24小时
                        try:
                            os.remove(filepath)
                            print(f'已清理过期文件: {filepath}')
                        except:
                            pass

# 在应用启动时运行清理
before_first_request_funcs = []
def before_first_request():
    cleanup_old_files()
    # 每小时清理一次
    import threading
    def cleanup_scheduler():
        while True:
            time.sleep(3600)  # 1小时
            cleanup_old_files()
    
    # 启动后台清理线程
    thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    thread.start()

# 添加到Flask的钩子
app.before_first_request(before_first_request)

if __name__ == '__main__':
    import time
    # 添加缺少的import
    print('启动人体姿态识别服务...')
    print(f'处理脚本路径: {PROCESS_VIDEO_PATH}')
    
    # 确保有临时目录的访问权限
    temp_dir = tempfile.gettempdir()
    print(f'临时目录: {temp_dir}')
    
    # 启动服务器
    print('服务器启动在: http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)