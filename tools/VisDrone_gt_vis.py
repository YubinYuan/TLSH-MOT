import cv2 
import pandas as pd 
import os 
import numpy as np

# 配置路径 

import cv2
import numpy as np
import os

def visualize_ground_truth(video_path, gt_path, output_dir):
    """
    可视化VisDrone数据集的ground truth
    
    Args:
        video_path: 视频序列帧的路径
        gt_path: ground truth文件的路径
        output_dir: 输出可视化结果的路径
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取ground truth文件
    # 格式: <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    gt_data = {}
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            frame_idx = int(line[0])
            if frame_idx not in gt_data:
                gt_data[frame_idx] = []
            
            # 获取边界框坐标
            bbox = [float(x) for x in line[2:6]]  # [left, top, width, height]
            target_id = int(line[1])
            gt_data[frame_idx].append((bbox, target_id))
    
    # 处理每一帧
    frame_files = sorted(os.listdir(video_path))
    for frame_idx, frame_file in enumerate(frame_files, 1):
        # 读取图像
        frame = cv2.imread(os.path.join(video_path, frame_file))
        if frame is None:
            continue
            
        # 在当前帧上绘制所有ground truth边界框
        if frame_idx in gt_data:
            for bbox, target_id in gt_data[frame_idx]:
                x, y, w, h = [int(v) for v in bbox]
                
                # 绘制边界框
                color = np.random.randint(0, 255, size=3).tolist()
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 绘制目标ID
                cv2.putText(frame, f'{target_id}', (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 在左上角绘制帧号
        cv2.putText(frame, f'Frame: {frame_idx}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(output_path, frame)
        
        print(f'Processed frame {frame_idx}')

# 使用示例
if __name__ == '__main__':
    video_path = r'J:\VisDrone2019-MOT-test-dev\sequences\uav0000088_00290_v'  # 视频帧所在文件夹
    gt_path = r'J:\VisDrone2019-MOT-test-dev\annotations\uav0000088_00290_v.txt'          # ground truth文件路径
    output_dir = r'J:\VisDrone2019-MOT-test-dev\hhh\uav0000088_00290_v'       # 输出文件夹
    
    visualize_ground_truth(video_path, gt_path, output_dir)
