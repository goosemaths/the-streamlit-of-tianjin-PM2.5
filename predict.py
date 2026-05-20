import os
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import torch.nn as nn

# --- 配置 (请根据实际修改) ---
MODEL_DIR = r"D:\vscodeproject\saved_models_20260427-0032"
PATH_METEO = r"D:\新下载\2025年11月~2026年3月天津市国控站点空气质量+气象小时数据.xlsx"
PATH_AIR = r"D:\新下载\国控站点空气质量小时数据_20251001.xlsx"
POI_FOLDER = r"D:\新下载\poi"

from process import AirNet, generate_base_grid, aggregate_pois_radius, build_spatial_mapping, preprocess_dynamic_data, POI_CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_prediction_with_explanation():
    # 1. 手动定义特征列表 (必须与训练脚本中的顺序严格一致)
    # 顺序：气象(7个) + 背景(1个) + POI(5个)
    FEATURES_LIST = [
        '气温', '气压', '相对湿度', '降水量', '风速', 'wind_u', 'wind_v', 
        'bg_pm25', 
        'cnt_industrial', 'cnt_commercial', 'cnt_nature', 'cnt_transport', 'cnt_catering'
    ]
    
    # 2. 加载标准化器
    scaler_x = joblib.load(os.path.join(MODEL_DIR, 'scaler_x.pkl'))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.pkl'))
    
    # 3. 加载模型 (使用定义的特征长度)
    models = []
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    print(f"检测到输入特征维度: {len(FEATURES_LIST)}")
    
    for f in model_files:
        m = AirNet(len(FEATURES_LIST)).to(device) # 这里改用 FEATURES_LIST 的长度
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f), map_location=device))
        m.eval()
        models.append(m)

    # 2. 生成地理基础数据 (一次性计算)
    grid_base = generate_base_grid()
    poi_static = aggregate_pois_radius(grid_base, POI_FOLDER, 2000)
    grid_mapped, _ = build_spatial_mapping(grid_base, PATH_METEO, PATH_AIR)
    
    # 【新增：保存静态属性字典表，用于后续解释】
    # 这张表包含了每个网格的坐标以及它周围 5 类 POI 的具体数量
    grid_info = pd.merge(grid_mapped[['grid_id', 'lng_wgs84', 'lat_wgs84', 'nearest_meteo_id']], 
                         poi_static, on='grid_id')
    grid_info.to_csv("grid_static_attributes.csv", index=False, encoding='utf-8-sig')
    print("✅ 静态网格特征表已导出 (grid_static_attributes.csv)，你可以用它分析 POI 影响。")

    # 3. 准备动态数据
    df_met_raw = pd.read_excel(PATH_METEO)
    df_air_raw = pd.read_excel(PATH_AIR)
    df_met_ts, df_air_ts = preprocess_dynamic_data(df_met_raw, df_air_raw)
    df_bg = df_air_ts.groupby('Datetime')['PM2p5'].mean().reset_index().rename(columns={'PM2p5': 'bg_pm25'})
    
    all_timestamps = sorted(df_bg['Datetime'].unique())
    results_list = []
    
    # 4. 预测循环
    for ts in tqdm(all_timestamps, desc="逐小时模拟中"):
        current_bg = df_bg[df_bg['Datetime'] == ts]['bg_pm25'].values[0]
        current_met = df_met_ts[df_met_ts['Datetime'] == ts]
        if current_met.empty:
            # print(f"跳过 {ts}: 该时刻气象数据完全缺失")
            continue
        # 对齐特征
        df_step = pd.merge(grid_info, current_met, left_on='nearest_meteo_id', right_on='Station_Id_C')
        if df_step.empty:
            # print(f"跳过 {ts}: 空间匹配后无有效数据（可能是对应气象站该时刻无值）")
            continue
        df_step['bg_pm25'] = current_bg
        
        X_raw = df_step[FEATURES_LIST].values.astype('float32')
        if X_raw.shape[0] == 0:
            continue
        X_scaled = torch.from_numpy(scaler_x.transform(X_raw)).to(device)
        
        with torch.no_grad():
            fold_preds = [m(X_scaled).cpu().numpy() for m in models]
            avg_pred = np.mean(fold_preds, axis=0)
            
        pm25_final = scaler_y.inverse_transform(avg_pred).flatten()
        
        # 存储动态预测结果
        results_list.append(pd.DataFrame({
            'dt': ts,
            'id': df_step['grid_id'], # 只存 ID，不存重复的 lng/lat 和 POI，节省空间
            'v': pm25_final.astype('float16') # 使用 float16 进一步压缩体积
        }))

    # 5. 合并并存储
    print("正在封装数据...")
    final_output = pd.concat(results_list)
    final_output.to_parquet("tianjin_pm25_predictions.parquet", compression='snappy')
    print("🚀 动态预测结果已保存。")

if __name__ == "__main__":
    run_prediction_with_explanation()