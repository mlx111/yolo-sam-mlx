#!/usr/bin/env python3
"""
GPU 加速版本的点云补全测试脚本
专为 RTX 5070 优化，充分利用 GPU 性能
"""

import torch
import numpy as np
import open3d as o3d
import sys
import os
import time
from pathlib import Path

# 添加 PoinTr 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'PoinTr'))

from models.PoinTr import PoinTr
from utils.config import cfg_from_yaml_file
from easydict import EasyDict

class GPUPointCloudCompleter:
    def __init__(self, ckpt_path, config_path=None, device='cuda'):
        # GPU 检测和优化设置
        self.device = self._setup_device(device)
        self._optimize_gpu_settings()
        
        # 1. 配置模型参数
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'PoinTr', 'cfgs', 'cfgs', 'ShapeNet55_models', 'PoinTr.yaml')
        
        print(f"加载配置文件: {config_path}")
        self.config = cfg_from_yaml_file(config_path)
        
        # 2. 初始化模型
        print(f"正在将模型加载到 {self.device.upper()}...")
        self.model = PoinTr(self.config.model).to(self.device)
        
        # 3. 加载权重
        print(f"加载模型权重: {ckpt_path}")
        self._load_checkpoint(ckpt_path)
        
        # 4. 设置为推理模式
        self.model.eval()
        
        # 5. 启用推理优化
        if self.device == 'cuda':
            self._enable_inference_optimizations()
        
        print("GPU 优化模型加载完成！")

    def _setup_device(self, device):
        """自动检测并设置最佳设备"""
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA 不可用，切换到 CPU")
            return 'cpu'
        elif device == 'cuda':
            # 检查 GPU 信息
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            cuda_version = torch.version.cuda
            
            print(f"检测到 GPU: {gpu_name}")
            print(f"GPU 内存: {gpu_memory:.1f} GB")
            print(f"CUDA 版本: {cuda_version}")
            
            # RTX 5070 特性检测
            if '5070' in gpu_name:
                print("✓ 检测到 RTX 5070，启用最佳优化设置")
            
            return 'cuda'
        
        return device

    def _optimize_gpu_settings(self):
        """优化 GPU 设置"""
        if self.device == 'cuda':
            # 启用 cuDNN benchmark 模式，加速卷积操作
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置内存分配策略
            torch.cuda.empty_cache()
            
            print("✓ GPU 优化设置已启用")

    def _enable_inference_optimizations(self):
        """启用推理优化"""
        # 编译模型以提升推理速度（PyTorch 2.0+）
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✓ 启用 torch.compile 加速")
            except Exception as e:
                print(f"torch.compile 失败，继续使用常规模式: {e}")
        
        # 启用混合精度
        self.use_amp = True
        print("✓ 启用自动混合精度 (AMP)")

    def _load_checkpoint(self, ckpt_path):
        """加载预训练权重"""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 处理不同的权重格式
        if 'model_state' in checkpoint:
            pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()}
        elif 'model' in checkpoint:
            pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        elif 'state_dict' in checkpoint:
            pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
        model_dict = self.model.state_dict()
        
        # 只加载匹配的权重
        matched_weights = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_weights[k] = v
                else:
                    print(f"形状不匹配: {k}, 模型: {model_dict[k].shape}, 权重: {v.shape}")
            else:
                print(f"未找到层: {k}")
        
        print(f"成功匹配 {len(matched_weights)}/{len(model_dict)} 层权重")
        model_dict.update(matched_weights)
        self.model.load_state_dict(model_dict)

    def normalize(self, points):
        """将点云归一化到单位球内"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        return points, centroid, max_dist

    def denormalize(self, points, centroid, scale):
        """将点云还原到原始尺度"""
        return points * scale + centroid

    def resample_points(self, points, target_n=2048):
        """重采样点云到目标数量"""
        current_n = len(points)
        
        if current_n == target_n:
            return points
        elif current_n > target_n:
            indices = np.random.choice(current_n, target_n, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(current_n, target_n, replace=True)
            return points[indices]

    def complete_pointcloud(self, partial_pcd, input_points=2048):
        """
        GPU 加速的点云补全
        
        Args:
            partial_pcd: (N, 3) numpy array, 输入的部分点云
            input_points: 输入点云数量
            
        Returns:
            complete_pcd: (M, 3) numpy array, 补全后的完整点云
            inference_time: 推理耗时（秒）
        """
        # 预处理计时开始
        preprocess_start = time.time()
        
        # 1. 预处理
        points = self.resample_points(partial_pcd, target_n=input_points)
        points_norm, centroid, scale = self.normalize(points)
        
        # 2. 转换为张量并移到 GPU
        inp = torch.from_numpy(points_norm).float().unsqueeze(0).to(self.device)  # (1, N, 3)
        inp = inp.transpose(2, 1).contiguous()  # (1, 3, N) PoinTr 需要的格式
        
        preprocess_time = time.time() - preprocess_start
        
        # 推理计时开始
        inference_start = time.time()
        
        # 3. GPU 加速推理
        with torch.no_grad():
            if self.device == 'cuda' and self.use_amp:
                # 使用自动混合精度
                with torch.cuda.amp.autocast():
                    ret = self.model(inp)
            else:
                ret = self.model(inp)
                
            coarse_points, fine_points = ret
            
            # 使用 CUDA Stream 异步处理
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # fine_points shape: (B, M, 3)
            pred_points = fine_points.squeeze(0).cpu().numpy()
        
        inference_time = time.time() - inference_start
        
        # 4. 后处理
        complete_points = self.denormalize(pred_points, centroid, scale)
        
        return complete_points, coarse_points.squeeze(0).cpu().numpy(), {
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'total_time': preprocess_time + inference_time
        }

def benchmark_gpu_vs_cpu(ckpt_path, test_data):
    """对比 GPU 和 CPU 性能"""
    print("\n" + "="*50)
    print("GPU vs CPU 性能对比测试")
    print("="*50)
    
    # GPU 测试
    print("\n🚀 GPU 测试...")
    gpu_completer = GPUPointCloudCompleter(ckpt_path, device='cuda')
    gpu_result, _, gpu_times = gpu_completer.complete_pointcloud(test_data)
    
    # CPU 测试
    print("\n🐌 CPU 测试...")
    cpu_completer = GPUPointCloudCompleter(ckpt_path, device='cpu')
    cpu_result, _, cpu_times = cpu_completer.complete_pointcloud(test_data)
    
    # 性能对比
    print("\n" + "="*50)
    print("性能对比结果")
    print("="*50)
    print(f"GPU 推理时间: {gpu_times['inference_time']:.3f} 秒")
    print(f"CPU 推理时间: {cpu_times['inference_time']:.3f} 秒")
    print(f"GPU 加速比: {cpu_times['inference_time']/gpu_times['inference_time']:.1f}x")
    print(f"GPU 总时间: {gpu_times['total_time']:.3f} 秒")
    print(f"CPU 总时间: {cpu_times['total_time']:.3f} 秒")
    
    return gpu_result, cpu_result

def main():
    """主函数"""
    # 配置参数
    ckpt_path = "pointr_training_from_scratch_c55_best.pth"
    input_file = "outputs/right_apple.npy"
    output_dir = "outputs"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎯 RTX 5070 GPU 加速点云补全测试")
    print("="*50)
    
    # 初始化 GPU 补全器
    try:
        completer = GPUPointCloudCompleter(ckpt_path)
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    # 加载或创建测试数据
    if os.path.exists(input_file):
        print(f"\n📂 加载输入点云: {input_file}")
        input_points = np.load(input_file)
    else:
        print(f"\n⚠️  未找到输入文件 {input_file}，创建示例数据...")
        # 创建示例苹果点云
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        points = np.asarray(pcd.points)
        mask = points[:, 2] > -0.1  # 只保留上半部分
        input_points = points[mask]
        np.save(input_file, input_points)
    
    print(f"输入点云数量: {len(input_points)}")
    
    # 执行点云补全
    print("\n🔧 开始点云补全...")
    try:
        complete_points, coarse_points, timing = completer.complete_pointcloud(input_points)
        print(f"✅ 补全完成！")
        print(f"   输出点云数量: {len(complete_points)}")
        print(f"   预处理时间: {timing['preprocess_time']:.3f} 秒")
        print(f"   推理时间: {timing['inference_time']:.3f} 秒")
        print(f"   总时间: {timing['total_time']:.3f} 秒")
    except Exception as e:
        print(f"❌ 补全失败: {e}")
        return
    
    # 保存结果
    pcd_complete = o3d.geometry.PointCloud()
    pcd_complete.points = o3d.utility.Vector3dVector(complete_points)
    o3d.io.write_point_cloud(os.path.join(output_dir, "gpu_complete_apple.ply"), pcd_complete)
    np.save(os.path.join(output_dir, "gpu_complete_apple.npy"), complete_points)
    print(f"💾 结果已保存到 {output_dir}")
    
    # 可选：性能对比
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_gpu_vs_cpu(ckpt_path, input_points)
    
    # 可视化结果
    print("\n👀 显示结果...")
    pcd_input = o3d.geometry.PointCloud()
    pcd_input.points = o3d.utility.Vector3dVector(input_points)
    pcd_input.paint_uniform_color([1, 0, 0])  # 红色
    
    pcd_complete_vis = o3d.geometry.PointCloud()
    pcd_complete_vis.points = o3d.utility.Vector3dVector(complete_points)
    pcd_complete_vis.paint_uniform_color([0, 1, 0])  # 绿色
    
    pcd_coarse = o3d.geometry.PointCloud()
    pcd_coarse.points = o3d.utility.Vector3dVector(coarse_points)
    pcd_coarse.paint_uniform_color([0, 0, 1])  # 蓝色
    
    o3d.visualization.draw_geometries(
        [pcd_input, pcd_complete_vis, pcd_coarse], 
        window_name="GPU 加速点云补全 (红:输入, 绿:补全, 蓝:粗糙)"
    )

if __name__ == "__main__":
    main()
