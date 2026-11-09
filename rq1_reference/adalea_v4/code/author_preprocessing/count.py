import os
import csv

def count_jpg_in_folders(root_folder):
    """
    统计根文件夹中每个子文件夹的JPG图像数量
    
    参数:
        root_folder: 要统计的根文件夹路径
        
    返回:
        包含文件夹名和JPG数量的列表，格式为[(文件夹名, JPG数量), ...]
    """
    results = []
    
    # 遍历根文件夹下的所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # 确保是文件夹而不是文件
        if os.path.isdir(folder_path):
            jpg_count = 0
            
            # 遍历子文件夹中的所有文件
            for filename in os.listdir(folder_path):
                # 检查文件是否是JPG图像（不区分大小写）
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    jpg_count += 1
            
            results.append((folder_name, jpg_count))
    
    return results

def save_to_csv(data, output_file):
    """
    将统计结果保存为CSV文件
    
    参数:
        data: 要保存的数据，格式为[(文件夹名, JPG数量), ...]
        output_file: 输出CSV文件路径
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Folder Name', 'JPG Count'])
        # 写入数据
        writer.writerows(data)

if __name__ == "__main__":
    # 设置要统计的文件夹路径
    root_folder = "train_raw_frames".strip()
    
    # 设置输出CSV文件路径
    output_csv = os.path.join(root_folder, 'jpg_count_results.csv')
    
    # 统计JPG数量
    results = count_jpg_in_folders(root_folder)
    
    # 保存为CSV
    save_to_csv(results, output_csv)
    
    print(f"统计完成，结果已保存到: {output_csv}")