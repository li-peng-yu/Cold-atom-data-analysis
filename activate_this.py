#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
冷原子吸收成像批量处理脚本 (v5 - 支持多文件夹批量处理)

功能:
1. 自动遍历 *指定* 的 'runXX' 文件夹。
2. 为每个 'runXX' 文件夹创建 *独立* 的 'runXX_results' 输出文件夹。
3. 计算光密度 (Optical Density, OD) 图像，并处理潜在的数学异常。
4. 将每张 OD 图像保存为伪彩色 .png 图像 (存入 output_images/)。
5. 使用高斯模糊 OD 图像来稳健地定位 ROI 中心。
6. 对每张 OD 图像在感兴趣区域 (ROI) 内拟合 2D 高斯函数。
7. 将拟合结果 (原始ROI, 拟合图, 残差图) 保存为 .png 图像 (存入 output_images/)。
8. 增强对 ROI 提取、数据内容和拟合参数的详细错误诊断。
9. 将 *每个 run* 的数据汇总并保存到 *各自* 的 CSV 文件 (runXX_results/runXX_results.csv)。

依赖库:
pip install numpy pandas opencv-python matplotlib scipy

如何运行:
1. 确保已安装上述 Python 库。
2. 确保所有 'runXX' 文件夹都在 'C:\\Users\\Administrator\\Desktop\\1028' 目录下。
3. 【重要】在脚本底部的 'if __name__ == "__main__":' 部分，
   确认 'BASE_PROJECT_DIR' 和 'RUN_NUMBERS' 列表正确无误。
4. 运行此脚本: python process_cold_atoms_v5.py
5. 每个 'runXX' 的处理结果将保存在对应的 'runXX_results' 文件夹中。
"""

import os
import cv2  # 使用 OpenCV 读取图像
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys
import traceback  # 用于打印完整的错误追溯

# --- 1. 配置参数 ---

# 【注意】文件夹路径已移至脚本底部的 __main__ 部分

# 计算参数
EPSILON = 1e-6  # 用于防止除以零或对负数取对数的小正数
ROI_SIZE = 500  # 感兴趣区域 (ROI) 的边长 (像素)
VIS_CMAP = 'hot'  # 可视化 OD 图像时使用的 Colormap (例如: 'hot', 'viridis', 'jet')
CROSS_SECTION_M2 = 8.5e-14
PIXEL_AREA_M2 = 6.7e-6

# ROI 中心定位高斯模糊核大小 (必须是奇数)
ROI_FINDING_BLUR_KSIZE = 51

# 拟合参数检查阈值
MIN_AMPLITUDE_FOR_FIT = 0.05
MAX_SIGMA_FACTOR = 0.8
MIN_SIGMA = 1.0


# --- 2. 辅助函数：2D 高斯模型 ---

def gaussian_2d(xy_data, A, x0, y0, sigma_x, sigma_y, C):
    """
    二维高斯函数模型 (用于 scipy.optimize.curve_fit)
    """
    x, y = xy_data
    sigma_x = max(sigma_x, EPSILON)
    sigma_y = max(sigma_y, EPSILON)
    exponent = -(((x - x0) ** 2 / (2 * sigma_x ** 2)) +
                 ((y - y0) ** 2 / (2 * sigma_y ** 2)))
    g = A * np.exp(exponent) + C
    return g.ravel()


# --- 3. 核心处理函数：处理单组图像 ---

def process_image_group(atom_path, probe_path, dark_path, output_image_dir):
    """
    加载、计算并分析一组 (Atoms, Probe, Dark) 图像。
    返回一个包含提取数据的字典。

    【V5 更改】: 增加了 'output_image_dir' 参数，以接收动态的输出路径。
    """

    # 初始化所有结果为 NaN，以防中途失败
    result_data = {
        'filename_atom': atom_path.name,
        'filename_probe': probe_path.name,
        'filename_dark': dark_path.name,
        'Summed_OD_ROI': np.nan,
        'Fit_Amplitude (A)': np.nan,
        'Fit_x_center': np.nan,
        'Fit_y_center': np.nan,
        'Fit_sigma_x': np.nan,
        'Fit_sigma_y': np.nan,
        'Fit_Offset (C)': np.nan,
        'ROI_y_min': np.nan,
        'ROI_y_max': np.nan,
        'ROI_x_min': np.nan,
        'ROI_x_max': np.nan,
        'ROI_center_y_guess': np.nan,
        'ROI_center_x_guess': np.nan,
    }

    try:
        # --- 功能 2: 加载图像并计算 OD ---
        img_atoms = cv2.imread(str(atom_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img_probe = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img_dark = cv2.imread(str(dark_path), cv2.IMREAD_GRAYSCALE).astype(np.float64)

        if img_atoms is None or img_probe is None or img_dark is None:
            print(f"  [警告] 无法加载组 {atom_path.name} 中的一张或多张图像。跳过此组。")
            return None

        numerator = img_atoms - img_dark
        denominator = img_probe - img_dark

        numerator[numerator <= 0] = EPSILON
        denominator[denominator <= 0] = EPSILON

        od_image = -np.log(numerator / denominator)
        od_image[~np.isfinite(od_image)] = 0.0

        # --- 功能 3: 输出 OD 可视化图片 ---
        # 【V5 更改】: 使用传入的 output_image_dir
        output_filename = output_image_dir / f"OD_{atom_path.stem}.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(od_image, cmap=VIS_CMAP)
        plt.colorbar(label='Optical Density (OD)')
        plt.title(f'OD: {atom_path.name}')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.savefig(output_filename)
        plt.close()

        # --- 功能 4: 提取并保存关键数据 ---

        # 4a. 定义 ROI
        (h, w) = od_image.shape

        od_image_blurred = cv2.GaussianBlur(od_image, (ROI_FINDING_BLUR_KSIZE, ROI_FINDING_BLUR_KSIZE), 0)
        y_center_px, x_center_px = np.unravel_index(np.argmax(od_image_blurred), (h, w))

        half_roi = ROI_SIZE // 2

        y_min = max(0, y_center_px - half_roi)
        y_max = min(h, y_center_px + half_roi)
        x_min = max(0, x_center_px - half_roi)
        x_max = min(w, x_center_px + half_roi)

        result_data['ROI_y_min'] = y_min
        result_data['ROI_y_max'] = y_max
        result_data['ROI_x_min'] = x_min
        result_data['ROI_x_max'] = x_max
        result_data['ROI_center_y_guess'] = y_center_px
        result_data['ROI_center_x_guess'] = x_center_px

        od_roi = od_image[y_min:y_max, x_min:x_max]

        if od_roi.size == 0:
            print(f"  [错误] 组 {atom_path.name} 无法提取有效的 ROI。")
            # ... (省略详细错误打印，原脚本中存在)
            return None

        if not np.any(np.isfinite(od_roi)):
            print(f"  [错误] 组 {atom_path.name} 的 ROI (shape: {od_roi.shape}) 只包含非有限值 (NaN/Inf)。跳过此组。")
            return None

        if np.all(od_roi == od_roi.flat[0]):
            print(
                f"  [错误] 组 {atom_path.name} 的 ROI (shape: {od_roi.shape}) 所有像素值都相同 ({od_roi.flat[0]})，无法进行高斯拟合。跳过此组。")
            return None

        if np.max(od_roi) - np.min(od_roi) < EPSILON * 100:
            print(
                f"  [警告] 组 {atom_path.name} 的 ROI (shape: {od_roi.shape}) 数据变化过小 (Max-Min < {EPSILON * 100:.2e})，高斯拟合可能无意义。")

        (h_roi, w_roi) = od_roi.shape

        # 4b. 计算 ROI 内的 OD 总和
        summed_od_roi = np.sum(od_roi[np.isfinite(od_roi)])
        result_data['Summed_OD_ROI'] = summed_od_roi

        # 4c. 准备拟合 2D 高斯函数
        y_roi, x_roi = np.indices(od_roi.shape)
        xy_data = np.vstack((x_roi.ravel(), y_roi.ravel()))
        z_data = od_roi.ravel()
        # 【V6 新增】: 根据 OD 总和计算原子数
        # N = (Pixel Area / Cross Section) * Summed_OD
        if CROSS_SECTION_M2 <= 0 or PIXEL_AREA_M2 <= 0:
            print(f"  [警告] 组 {atom_path.name}：散射截面或像素面积未正确设置(<=0)，无法计算原子数。")
            atom_number = np.nan
        else:
            atom_number = (PIXEL_AREA_M2 / CROSS_SECTION_M2) * summed_od_roi

        result_data['Atom_Number'] = atom_number

        A_guess = np.max(od_roi) - np.min(od_roi)
        y0_guess_roi, x0_guess_roi = np.unravel_index(np.argmax(od_roi), od_roi.shape)
        sigma_x_guess = w_roi / 4.0
        sigma_y_guess = h_roi / 4.0
        C_guess = np.min(od_roi)

        p0 = [A_guess, x0_guess_roi, y0_guess_roi, sigma_x_guess, sigma_y_guess, C_guess]

        # 4d. 执行拟合
        try:
            bounds = (
                [0, 0, 0, MIN_SIGMA, MIN_SIGMA, -np.inf],
                [np.inf, w_roi, h_roi, w_roi * MAX_SIGMA_FACTOR, h_roi * MAX_SIGMA_FACTOR, np.inf]
            )
            popt, pcov = curve_fit(gaussian_2d, xy_data, z_data, p0=p0, bounds=bounds)

            A_fit, x0_fit_roi, y0_fit_roi, sigma_x_fit, sigma_y_fit, C_fit = popt

            if A_fit <= MIN_AMPLITUDE_FOR_FIT:
                print(
                    f"  [警告] 组 {atom_path.name} 拟合振幅过小 ({A_fit:.4f} <= {MIN_AMPLITUDE_FOR_FIT:.4f})，拟合无效。")
                raise RuntimeError("Fit amplitude too small.")
            if sigma_x_fit < MIN_SIGMA or sigma_y_fit < MIN_SIGMA:
                print(
                    f"  [警告] 组 {atom_path.name} 拟合宽度过小 (sigma_x={sigma_x_fit:.2f}, sigma_y={sigma_y_fit:.2f})，拟合无效。")
                raise RuntimeError("Fit sigma too small.")

            x_center_fit_global = x0_fit_roi + x_min
            y_center_fit_global = y0_fit_roi + y_min
            sigma_x_fit = np.abs(sigma_x_fit)
            sigma_y_fit = np.abs(sigma_y_fit)

            result_data['Fit_Amplitude (A)'] = A_fit
            result_data['Fit_x_center'] = x_center_fit_global
            result_data['Fit_y_center'] = y_center_fit_global
            result_data['Fit_sigma_x'] = sigma_x_fit
            result_data['Fit_sigma_y'] = sigma_y_fit
            result_data['Fit_Offset (C)'] = C_fit

            # --- 可视化 2D 高斯拟合结果 ---
            try:
                fit_z_data = gaussian_2d(xy_data, *popt)
                fit_image = fit_z_data.reshape(od_roi.shape)
                residuals = od_roi - fit_image

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
                fig.suptitle(f'Gaussian Fit Analysis: {atom_path.name}', fontsize=16)

                vmin = np.min(od_roi)
                vmax = np.max(od_roi)

                im1 = ax1.imshow(od_roi, cmap=VIS_CMAP, origin='lower',
                                 extent=[0, w_roi, 0, h_roi], vmin=vmin, vmax=vmax)
                atom_num_str = f"{result_data.get('Atom_Number', np.nan):.2e}"
                ax1.set_title(f'Original OD ROI\n(Sum: {summed_od_roi:.2f} | Atoms: {atom_num_str})')
                ax1.set_xlabel('ROI X Pixel')
                ax1.set_ylabel('ROI Y Pixel')
                fig.colorbar(im1, ax=ax1, shrink=0.8)

                im2 = ax2.imshow(fit_image, cmap=VIS_CMAP, origin='lower',
                                 extent=[0, w_roi, 0, h_roi], vmin=vmin, vmax=vmax)
                ax2.set_title(f'2D Gaussian Fit (A={A_fit:.2f}, sx={sigma_x_fit:.2f}, sy={sigma_y_fit:.2f})')
                ax2.set_xlabel('ROI X Pixel')
                fig.colorbar(im2, ax=ax2, shrink=0.8)

                res_max_abs = np.nanmax(np.abs(residuals[np.isfinite(residuals)])) if np.any(
                    np.isfinite(residuals)) else 1.0
                im3 = ax3.imshow(residuals, cmap='coolwarm', origin='lower',
                                 extent=[0, w_roi, 0, h_roi],
                                 vmin=-res_max_abs, vmax=res_max_abs)
                ax3.set_title('Residuals (Data - Fit)')
                ax3.set_xlabel('ROI X Pixel')
                fig.colorbar(im3, ax=ax3, shrink=0.8)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # 【V5 更改】: 使用传入的 output_image_dir
                fit_plot_filename = output_image_dir / f"FIT_{atom_path.stem}.png"
                plt.savefig(fit_plot_filename)
                plt.close(fig)

            except Exception as plot_e:
                print(f"  [警告] 组 {atom_path.name} 的拟合结果可视化失败: {plot_e}")
                plt.close()

        except (RuntimeError, ValueError) as e:
            print(f"  [警告] 组 {atom_path.name} 的 2D 高斯拟合失败: {e}")

        return result_data

    except Exception as e:
        print(f"  [严重错误] 处理组 {atom_path.name} 时发生意外错误。")
        traceback.print_exc()
        return None

def process_OD():
# --- 4. 主执行函数 ---

def process_run_folder(run_input_dir, run_output_dir):
    """
    【V5 新增】:
    此函数源自原版的 main() 函数。
    它负责处理 *一个* 指定的 'run' 文件夹，并将结果保存到 *一个* 指定的输出文件夹。
    """

    # --- 动态定义路径 ---
    INPUT_DIR = Path(run_input_dir)
    run_name = INPUT_DIR.name  # 例如 "run01"

    # 将此 run 的所有输出（图像、csv）放入一个专用的结果文件夹
    OUTPUT_BASE_DIR = Path(run_output_dir)
    OUTPUT_IMAGE_DIR = OUTPUT_BASE_DIR / "images"  # 例如 .../run01_results/images/
    OUTPUT_CSV = OUTPUT_BASE_DIR / f"{run_name}_results.csv"  # 例如 .../run01_results/run01_results.csv

    if not INPUT_DIR.exists():
        print(f"  错误: 输入文件夹 '{INPUT_DIR}' 不存在。跳过此文件夹。")
        return

    # 为此 run 创建输出文件夹
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  输入: {INPUT_DIR.resolve()}")
    print(f"  图像输出: {OUTPUT_IMAGE_DIR.resolve()}")
    print(f"  数据输出: {OUTPUT_CSV.resolve()}")

    image_files = sorted(list(INPUT_DIR.glob('T*_*.png')))

    if len(image_files) == 0:
        print(f"  错误: 在文件夹 {INPUT_DIR.name} 中未找到 'T*_*.png' 格式的 .png 文件。跳过此文件夹。")
        return

    if len(image_files) % 3 != 0:
        print(f"  警告: 找到 {len(image_files)} 个文件，不是 3 的倍数。")
        print("  最后一组不完整的图像将被忽略。")

    all_results = []

    for i in range(0, len(image_files) - 2, 3):
        atom_path = image_files[i]
        probe_path = image_files[i + 1]
        dark_path = image_files[i + 2]

        print(f"\n  [处理第 {i // 3 + 1} 组] Atoms: {atom_path.name}")

        # 【V5 更改】: 将 OUTPUT_IMAGE_DIR 路径传入
        result = process_image_group(atom_path, probe_path, dark_path, OUTPUT_IMAGE_DIR)

        if result:
            all_results.append(result)
            if not np.isnan(result.get('Fit_Amplitude (A)', np.nan)):
                print(f"    [成功] 组 {atom_path.name} 处理完毕，高斯拟合成功。")
            else:
                print(f"    [完成] 组 {atom_path.name} 处理完毕，但高斯拟合失败或参数不合理。")
        else:
            print(f"    [跳过] 组 {atom_path.name} 因严重错误跳过。")

    if not all_results:
        print(f"\n  {run_name} 处理完成，但没有成功提取任何数据。")
        return

    print(f"\n  --- {run_name} 处理完成 ---")
    print(f"  总共处理了 {len(all_results)} 组图像。")

    df = pd.DataFrame(all_results)

    columns_order = [
        'filename_atom', 'Summed_OD_ROI', 'Fit_Amplitude (A)',
        'Fit_x_center', 'Fit_y_center', 'Fit_sigma_x', 'Fit_sigma_y',
        'Fit_Offset (C)', 'filename_probe', 'filename_dark',
        'ROI_y_min', 'ROI_y_max', 'ROI_x_min', 'ROI_x_max',
        'ROI_center_y_guess', 'ROI_center_x_guess'
    ]
    final_columns = [col for col in columns_order if col in df.columns]
    df = df[final_columns]

    try:
        df.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')
        print(f"  所有数据已成功保存到: {OUTPUT_CSV}")
    except Exception as e:
        print(f"  错误: 无法保存 CSV 文件: {e}")


# --- 5. 【V5 新增】: 脚本主入口 ---

if __name__ == "__main__":

    # --- 请在这里配置您的路径和文件夹列表 ---

    # 1. 定义包含所有 'runXX' 文件夹的根目录
    #    【重要】使用 'r' 前缀来处理 Windows 路径中的反斜杠
    BASE_PROJECT_DIR = Path(r"C:\Users\Administrator\Desktop\1028")

    # 2. 定义您想要处理的 run 编号列表
    RUN_NUMBERS = [1,3,4,9,10,11,13,14,15,16,18]

    # ----------------------------------------

    print(f"--- 冷原子图像批量处理开始 (v5 - 多文件夹) ---")
    print(f"基础目录: {BASE_PROJECT_DIR}")
    print(f"待处理的 Run 编号: {RUN_NUMBERS}")

    total_runs = len(RUN_NUMBERS)

    for i, run_num in enumerate(RUN_NUMBERS):
        # 格式化文件夹名称，例如 1 -> "run01", 13 -> "run13"
        run_name = f"run{run_num:02d}"

        # 定义此 run 的输入和输出路径
        run_input_dir = BASE_PROJECT_DIR / run_name
        run_output_dir = BASE_PROJECT_DIR / f"{run_name}_results"

        print(f"\n=================================================")
        print(f"====== 开始处理文件夹 ({i + 1}/{total_runs}): {run_name} ======")
        print(f"=================================================")

        # 调用处理函数
        process_run_folder(run_input_dir, run_output_dir)

    print("\n\n--- 所有指定的文件夹均已处理完毕 ---")

    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import re
    import numpy as np

    # --- 1. 自动检测列名 ---
    PROFILE_1 = {'A': 'AmplitFit', 'sx': 'sigmaFit_x', 'sy': 'sigmaFit_y'}
    PROFILE_2 = {'A': 'Fit_Amplitude (A)', 'sx': 'Fit_sigma_x', 'sy': 'Fit_sigma_y'}
    PROFILE_3 = {'A': 'AmpliFit', 'sx': 'Fit_sigma_x', 'sy': 'Fit_sigma_y'}

    COLUMN_PROFILES = [PROFILE_1, PROFILE_2, PROFILE_3]

    # --- 2. 设置文件路径 ---
    DATA_DIR = r"C:\Users\xuanl\Desktop\data"
    OUTPUT_DIR = r"D:\Generated_Total_Atom_Number_Plots"

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    # --- 3. 辅助函数：选择正确的列名配置 ---
    def get_column_profile(csv_columns):
        """
        检查CSV文件的列名，返回一个匹配的配置 (profile)
        """
        for profile in COLUMN_PROFILES:
            required_cols = [profile['A'], profile['sx'], profile['sy']]
            if all(col in csv_columns for col in required_cols):
                return profile
        return None

        # --- 4. 主处理逻辑 ---


    def generate_all_plots():

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"输出文件夹已准备好: {OUTPUT_DIR}")
        except Exception as e:
            print(f"!!! 创建文件夹失败 !!!: {e}")
            return

        file_indices = range(1, 19)
        print(f"开始批量处理 {len(file_indices)} 个文件...")
        print(f"将绘制 总原子数 (N = 2*pi*A*sx*sy)。")
        print(f"将读取每个文件的 *所有* 行数据。")
        print(f"!!! Y轴将 *强制* 使用科学计数法。!!!")  # <--- 已更新
        print("--------------------------------------------------")

        plots_generated = 0
        files_failed = 0

        for m in file_indices:

            file_name = f'results_{m}.csv'
            file_path = os.path.join(DATA_DIR, file_name)

            if not os.path.exists(file_path):
                print(f"文件 {file_name} 未找到，跳过。")
                files_failed += 1
                continue

            print(f"正在处理文件: {file_name} ...")

            try:
                # 1. 读取 CSV (读取所有行)
                data_frame = pd.read_csv(file_path)

                if data_frame.empty:
                    print(f"  文件 {file_name} 为空，跳过。")
                    files_failed += 1
                    continue

                # 2. 自动检测列名
                profile = get_column_profile(data_frame.columns)

                if profile is None:
                    print(f"  !!! 错误 !!! 文件 {file_name} 的列名不匹配任何已知配置。")
                    files_failed += 1
                    continue

                x_values = []
                y_values_N = []

                # 3. 遍历 *所有* 行
                for index, row in data_frame.iterrows():
                    x_val = index + 1

                    amplitude = row[profile['A']]
                    sigma_x = row[profile['sx']]
                    sigma_y = row[profile['sy']]

                    total_atoms_N = 2 * np.pi * amplitude * sigma_x * sigma_y

                    x_values.append(x_val)
                    y_values_N.append(total_atoms_N)

                # 4. 生成图表
                if not y_values_N:
                    print(f"  在 {file_name} 中未提取到有效数据，跳过。")
                    files_failed += 1
                    continue

                plt.figure(figsize=(12, 7))
                plt.plot(x_values, y_values_N, marker='o', linestyle='-')

                plt.xlabel('数据点序号 (Index)')
                plt.ylabel('总原子数 (N)')
                plt.title(f'总原子数变化 (数据源: {file_name})')

                # --- 5. (已修改) 强制 Y 轴使用科学计数法 ---
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                # ----------------------------------------

                plt.grid(True)
                plt.tight_layout()

                output_filename = f'plot_total_atoms_results_{m}.png'
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                plt.savefig(output_path)
                plt.close()

                print(f"  成功 (共 {len(x_values)} 个点) (使用配置: {profile['A']}) -> {output_filename}")
                plots_generated += 1

            except Exception as e:
                print(f"  !!! 处理文件 {file_name} 时发生未知错误 !!!: {e}")
                files_failed += 1

        # --- 6. 最终总结 ---
        print("--------------------------------------------------")
        print("批量处理完成。")
        print(f"成功生成图表: {plots_generated} 张")
        print(f"失败或跳过文件: {files_failed} 个")
        print(f"所有图表已保存到: {OUTPUT_DIR}")


    # --- 运行主程序 ---
    if __name__ == "__main__":
        generate_all_plots()

