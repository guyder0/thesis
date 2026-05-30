import torch
import torch.nn.functional as F
import numpy as np
import lpips

from skimage.metrics import structural_similarity as ssim_metric
from plyfile import PlyData
from gsplat.rendering import rasterization

def load_ply(path, device="cuda"):
    print(f"Загрузка PLY файла: {path}...")
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # Координаты центров (means)
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    
    # Масштабы (scales) хранятся в логарифмическом виде
    scales_raw = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
    scales = torch.exp(torch.tensor(scales_raw, dtype=torch.float32, device=device))
    
    # 3Повороты (quaternions) в формате [w, x, y, z]
    quats_raw = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=-1)
    quats = torch.tensor(quats_raw, dtype=torch.float32, device=device)
    quats = F.normalize(quats, p=2, dim=-1) # Нормализуем кватернионы
    
    # Прозрачность (opacities) хранится в пространстве логитов
    opacities_raw = vertex['opacity']
    opacities = torch.sigmoid(torch.tensor(opacities_raw, dtype=torch.float32, device=device))
    
    # Цвета и сферические гармоники (SH)
    # Базовый цвет (f_dc) имеет форму [N, 3] (R, G, B)
    f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    sh0 = torch.tensor(f_dc, dtype=torch.float32, device=device).unsqueeze(1) # [N, 1, 3]
    
    # Высокочастотные коэффициенты гармоник (f_rest)
    sh_names = sorted(
        [name for name in vertex.data.dtype.names if name.startswith("f_rest_")],
        key=lambda x: int(x.split('_')[2])
    )
    
    f_rest = np.stack([vertex[name] for name in sh_names], axis=-1)
    f_rest_t = torch.tensor(f_rest, dtype=torch.float32, device=device)
    # 45 коэффициентов упорядочены по каналам: 15 для R, 15 для G, 15 для B.
    # Пересобираем в тензор формы [N, 15, 3]
    shN = f_rest_t.reshape(-1, 3, 15).transpose(1, 2) 
    colors = torch.cat([sh0, shN], dim=1) # Итоговая форма [N, 16, 3]
    sh_degree = 3
        
    print(f"Успешно загружено {len(means)} гауссиан.")
    return means, scales, quats, opacities, colors, sh_degree


def get_view_matrix(camera_pos, target, world_up=torch.tensor([0.0, 0.0, 1.0])):
    """
    Создает матрицу вида World-to-Camera (W2C) в системе координат OpenCV.
    """
    # Ось Z направлена ВПЕРЕД (от камеры к цели)
    z_axis = F.normalize(target - camera_pos, p=2, dim=0)
    
    # Ось X направлена ВПРАВО (векторное произведение Z и world_up)
    x_axis = F.normalize(torch.cross(z_axis, world_up.to(camera_pos.device), dim=0), p=2, dim=0)
    
    # Ось Y направлена ВНИЗ (векторное произведение Z и X)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    
    # Собираем матрицу поворота R (строки матрицы — это направления осей камеры)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0) # [3, 3]
    t = -R @ camera_pos # [3]
    
    viewmat = torch.eye(4, dtype=torch.float32, device=camera_pos.device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


def render_scene(ply_path, camera_pos, background=[0, 0, 0]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Не найден cuda")
        return
        
    means, scales, quats, opacities, colors, sh_degree = load_ply(ply_path, device=device)

    # Соответствует рендеру Blender
    width, height = 800, 1200
    focal = 1200

    K = torch.tensor([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)
    
    centroid = torch.zeros(3, dtype=torch.float32, device=device)
    camera_pos = torch.tensor(camera_pos, dtype=torch.float32, device=device)
    
    viewmat = get_view_matrix(camera_pos, centroid)
    
    # gsplat ожидает батч камер, поэтому добавляем размерность батча [C, ...]
    viewmats = viewmat.unsqueeze(0) # [1, 4, 4]
    Ks = K.unsqueeze(0) # [1, 3, 3]
    
    # Растеризация
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        backgrounds=torch.tensor(background, dtype=torch.float32, device=device) 
    )
    
    # Извлекаем изображение из батча и переносим на CPU
    # render_colors имеет форму [1, height, width, 3]
    img_tensor = render_colors[0].clamp(0.0, 1.0)
    img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    
    return img_np


import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_metric

# Инициализируем модель LPIPS (обычно используют на базе VGG)
# При первом запуске скрипт скачает веса сети
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()

def calculate_psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(gt, pred):
    return ssim_metric(gt, pred, channel_axis=2, data_range=1.0)

def calculate_lpips(gt, pred):
    # Переводим в PyTorch тензоры и меняем оси на [C, H, W]
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    
    # Нормализуем диапазон из [0, 1] в [-1, 1], как требует библиотека lpips
    gt_tensor = gt_tensor * 2.0 - 1.0
    pred_tensor = pred_tensor * 2.0 - 1.0

    gt_tensor = gt_tensor.to(device)
    pred_tensor = pred_tensor.to(device)
    
    # Считаем метрику без расчета градиентов
    with torch.no_grad():
        lpips_value = loss_fn_vgg(gt_tensor, pred_tensor)
        
    return lpips_value.item()

def evaluate_metrics(gt, pred):
    # Приводим к типу float32 и диапазону [0.0, 1.0]
    gt = gt.astype(np.float32) / 255.0
    pred = pred.astype(np.float32) / 255.0
    
    # Расчет метрик
    psnr_val = calculate_psnr(gt, pred)
    ssim_val = calculate_ssim(gt, pred)
    lpips_val = calculate_lpips(gt, pred)
    
    return f"PSNR {psnr_val:.4f}\nSSIM {ssim_val:.4f}\nLPIPS {lpips_val:.4f}"