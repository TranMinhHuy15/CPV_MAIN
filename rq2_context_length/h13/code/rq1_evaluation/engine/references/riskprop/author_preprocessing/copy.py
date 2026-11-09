import os
import shutil
from pathlib import Path

def copy_last_30_images(source_dir, target_dir):
    """
    复制每个子文件夹中最后30张图像到新目录，保留原始目录结构
    
    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    # 遍历源目录中的所有子文件夹
    for root, dirs, files in os.walk(source_dir):
        # 获取相对于源目录的路径
        relative_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, relative_path)
        
        # 创建对应的目标子目录
        Path(target_subdir).mkdir(parents=True, exist_ok=True)
        
        # 筛选出图片文件并按修改时间排序
        image_files = [f for f in files if f.lower().endswith(image_extensions)]
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)))
        
        # 获取最后30张图片
        last_30_images = image_files[-15:] if len(image_files) > 15 else image_files
        
        # 复制图片到目标目录
        for image in last_30_images:
            src_path = os.path.join(root, image)
            dst_path = os.path.join(target_subdir, image)
            shutil.copy2(src_path, dst_path)  # copy2会保留元数据
            
        print(f"已从 {relative_path} 复制 {len(last_30_images)} 张图片到 {target_subdir}")

# 使用示例
source_directory = "test_raw_frames"  # 替换为你的源目录路径
target_directory = "test_raw_frames_15"  # 替换为目标目录路径

copy_last_30_images(source_directory, target_directory)