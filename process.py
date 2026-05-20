import os
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
import osmnx as ox
import joblib
# --- 1. 配置区 ---
RADIUS_R = 2000   # 超参数：POI 影响半径 (米)
GRID_SIZE = 1000  # 网格大小 (米)
CITY_NAME = "Tianjin, China"

# 路径配置 (请确保路径正确)
FOLDER_POI = r"D:\新下载\poi"#poi文件夹
PATH_METEO = r"D:\新下载\2025年11月~2026年3月天津市国控站点空气质量+气象小时数据.xlsx"#气象数据
PATH_AIR = r"D:\新下载\国控站点空气质量小时数据_20251001.xlsx"#空气污染数据

POI_CONFIGS = {
    'cnt_industrial': ['1701', '1702', '1703'],   # 工厂企业
    'cnt_commercial': ['06', '14'],               # 购物、科教文化
    'cnt_nature': ['11', '1704'],                 # 风景名胜、绿色农业
    'cnt_transport': ['15'],                      # 交通设施
    'cnt_catering': ['05']                        # 餐饮服务
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# --- 2. 基础地理网格生成 ---
def generate_base_grid():
    print("正在生成天津市 1km 标准网格...")
    city = ox.geocode_to_gdf(CITY_NAME).to_crs(epsg=3857)
    minx, miny, maxx, maxy = city.total_bounds
    x_coords = np.arange(minx, maxx, GRID_SIZE)
    y_coords = np.arange(miny, maxy, GRID_SIZE)
    grid_cells = [Polygon([(x, y), (x+GRID_SIZE, y), (x+GRID_SIZE, y+GRID_SIZE), (x, y+GRID_SIZE)]) 
                  for x in x_coords for y in y_coords]
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=3857)
    grid = gpd.sjoin(grid, city, how="inner", predicate="intersects").reset_index(drop=True)
    grid['grid_id'] = [f"G{i:05d}" for i in grid.index]
    
    # 辅助坐标：WGS84 中心点用于匹配气象站
    grid_wgs84 = grid.to_crs(epsg=4326)
    grid['lng_wgs84'] = grid_wgs84.geometry.centroid.x
    grid['lat_wgs84'] = grid_wgs84.geometry.centroid.y
    return grid

# --- 3. POI 半径聚合 (静态特征) ---
def aggregate_pois_radius(grid_gdf, folder_path, r_meters):
    print(f"正在聚合 POI 特征 (半径 r={r_meters}m)...")
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(('.xlsx', '.csv')):
            path = os.path.join(folder_path, file)
            df = pd.read_excel(path) if file.endswith('.xlsx') else pd.read_csv(path)
            if 'typecode' in df.columns:
                df['typecode'] = df['typecode'].astype(str).str.zfill(6)
                all_dfs.append(df[['wgs84经度', 'wgs84纬度', 'typecode']])
    
    big_poi = pd.concat(all_dfs, ignore_index=True)
    gdf_poi = gpd.GeoDataFrame(big_poi, geometry=gpd.points_from_xy(big_poi['wgs84经度'], big_poi['wgs84纬度']), crs=4326).to_crs(epsg=3857)

    grid_centers = np.array(list(zip(grid_gdf.geometry.centroid.x, grid_gdf.geometry.centroid.y)))
    result = pd.DataFrame({'grid_id': grid_gdf['grid_id']})

    for cat, codes in POI_CONFIGS.items():
        pattern = '^(' + '|'.join(codes) + ')'
        cat_pois = gdf_poi[gdf_poi['typecode'].str.contains(pattern, na=False)]
        if not cat_pois.empty:
            tree = cKDTree(np.array(list(zip(cat_pois.geometry.x, cat_pois.geometry.y))))
            result[cat] = tree.query_ball_point(grid_centers, r=r_meters, return_length=True)
        else:
            result[cat] = 0
    return result
#预处理风向以及污染物的空值问题
def preprocess_dynamic_data(df_met, df_air):
    """
    预处理气象和空气质量动态数据
    """
    print("正在清洗动态数据 (处理风向与缺失值)...")
    
    # --- 1. 处理风向 ---
    # 将"静风"替换为 0，并转为数值
    df_met['风向'] = pd.to_numeric(df_met['风向'].replace('静风', 0), errors='coerce').fillna(0)
    # 将角度转为弧度
    rad = np.deg2rad(df_met['风向'])
    # 将风分解为 U (东西向) 和 V (南北向) 分量
    # u = speed * cos(theta), v = speed * sin(theta)
    df_met['wind_u'] = df_met['风速'] * np.cos(rad)
    df_met['wind_v'] = df_met['风速'] * np.sin(rad)
    
    # --- 2. 处理 PM2.5 缺失值 ---
    # 强制转为数值，非数字转为 NaN
    df_air['PM2p5'] = pd.to_numeric(df_air['PM2p5'], errors='coerce')
    # 按站点(Code)进行线性插值，填充短时间的缺失
    df_air['PM2p5'] = df_air.groupby('Code')['PM2p5'].transform(
        lambda x: x.interpolate(method='linear').ffill().bfill()
    )
    
    # 删除插值后依然无法填充的行 (比如该站整段缺失)
    df_air = df_air.dropna(subset=['PM2p5'])
    
    return df_met, df_air


# --- 4. 空间映射 (气象匹配 + 训练网格锁定) ---
def build_spatial_mapping(grid_gdf, meteo_path, air_path):
    print("建立全局气象映射与监测站网格锁定...")
    df_meteo_meta = pd.read_excel(meteo_path)[['Station_Id_C', 'Lon', 'Lat']].drop_duplicates()
    df_air_meta = pd.read_excel(air_path)[['Code', 'Lon', 'Lat']].drop_duplicates()
    
    # 全局气象匹配 (2万个网格)
    meteo_tree = cKDTree(df_meteo_meta[['Lon', 'Lat']].values)
    _, indices = meteo_tree.query(grid_gdf[['lng_wgs84', 'lat_wgs84']].values)
    grid_gdf['nearest_meteo_id'] = df_meteo_meta.iloc[indices]['Station_Id_C'].values
    
    # 锁定训练网格
    gdf_air = gpd.GeoDataFrame(df_air_meta, geometry=gpd.points_from_xy(df_air_meta['Lon'], df_air_meta['Lat']), crs=4326).to_crs(epsg=3857)
    air_in_grid = gpd.sjoin(gdf_air, grid_gdf[['grid_id', 'geometry']], how='inner', predicate='within')
    train_mapping = air_in_grid[['grid_id', 'Code']]
    
    return grid_gdf, train_mapping

# --- 5. 构建 LOSO 训练数据集 ---
def prepare_final_dataset(train_mapping, grid_mapped, poi_static, meteo_path, air_path):
    print("正在构建时空对齐训练集...")
    df_met_raw = pd.read_excel(meteo_path)
    df_air_raw = pd.read_excel(air_path) 
# --- 调用清洗函数 ---
    df_met_ts, df_air_ts = preprocess_dynamic_data(df_met_raw, df_air_raw)
     # 计算每小时全市平均 PM2.5 作为“背景浓度”
    df_bg = df_air_ts.groupby('Datetime')['PM2p5'].mean().reset_index().rename(columns={'PM2p5': 'bg_pm25'})
    df_air_ts = pd.merge(df_air_ts, df_bg, on='Datetime')
    samples = []
    for _, row in train_mapping.iterrows():
        g_id, a_code = row['grid_id'], row['Code']
        
        # 提取数据并对齐时间
        a_data = df_air_ts[df_air_ts['Code'] == a_code]
        m_id = grid_mapped[grid_mapped['grid_id'] == g_id]['nearest_meteo_id'].values[0]
        m_data = df_met_ts[df_met_ts['Station_Id_C'] == m_id]
        
        merged = pd.merge(a_data, m_data, on='Datetime')
        
        # 注入半径 POI 特征
        poi_vals = poi_static[poi_static['grid_id'] == g_id].iloc[0]
        for cat in POI_CONFIGS.keys():
            merged[cat] = poi_vals[cat]
            
        merged['grid_id'] = g_id
        samples.append(merged)
    final_df = pd.concat(samples, ignore_index=True)
    # --- 核心清理逻辑 ---
    # 1. 确保所有列都是数值型，非法字符转 NaN
    cols_to_check = ['PM2p5', '气温', '气压', '相对湿度', '降水量', '风速', 'wind_u', 'wind_v']
    for col in cols_to_check:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # 2. 彻底删除任何含有 NaN 的行
    before_count = len(final_df)
    final_df = final_df.dropna(subset=cols_to_check)
    
    # 3. 再次强制物理过滤
    final_df = final_df[final_df['PM2p5'] <= 500] # 剔除大于500的极端离群值
    
    print(f"清理完成：由于缺失值或极端值，剔除了 {before_count - len(final_df)} 行数据")
    return final_df

# --- 6. 神经网络模型与 TensorBoard 训练 ---
class AirNet(nn.Module):
    def __init__(self, input_dim):
        super(AirNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 减小第一层规模，防止过强记忆
            nn.ReLU(),
            nn.Dropout(0.2),         # 丢弃 20% 的神经元
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)
def train_loso(df):
    # 检查训练集中的极值
    print("--- 正在进行全量数据安全性审计与清洗 ---")
    # 1. 物理阈值过滤
    initial_len = len(df)
    df = df[(df['PM2p5'] >= 0) & (df['PM2p5'] <= 500)]
    
    # 2. 剔除样本量太少的“僵尸站”
    station_counts = df.groupby('grid_id').size()
    valid_stations = station_counts[station_counts > 48].index  # 至少有48小时数据的站才留着
    df = df[df['grid_id'].isin(valid_stations)]
    
    print(f"审计完成：剔除了 {initial_len - len(df)} 行异常数据，剩余有效监测站: {df['grid_id'].nunique()} 个")
    
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    save_dir = f"saved_models_{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 特征选择 (气象字段请根据你 Excel 实际列名修改)
    met_cols = ['气温', '气压', '相对湿度', '降水量', '风速', 'wind_u', 'wind_v','bg_pm25'] 
    poi_cols = list(POI_CONFIGS.keys())
    features = met_cols + poi_cols
    target = 'PM2p5' # 假设拟合 PM2.5
    df = df.dropna(subset=features + [target])
    X_raw = df[features].values.astype('float32')
    y_raw = df[target].values.reshape(-1, 1).astype('float32')
    groups = df['grid_id'].values

    # 标准化
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)
    joblib.dump(scaler_x, os.path.join(save_dir, 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
    logo = LeaveOneGroupOut()
    writer = SummaryWriter(f"logs/GPU_MSE_{run_id}")

    print(f"开始(GPU)留一站交叉验证 (站数: {logo.get_n_splits(groups=groups)})")
    
    for fold, (train_idx, val_idx) in enumerate(logo.split(X_scaled, y_scaled, groups)):
        val_station = groups[val_idx[0]]
        print(f"Fold {fold+1}: 留出站点 {val_station}")
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_scaled[train_idx]), torch.from_numpy(y_scaled[train_idx])), batch_size=4096, shuffle=True)
        val_x = torch.from_numpy(X_scaled[val_idx]).to(device)
        val_y = torch.from_numpy(y_scaled[val_idx]).to(device)
        model = AirNet(len(features)).to(device)
        opt = optim.Adam(model.parameters(), lr=0.0005,weight_decay=1e-4)
        crit = nn.HuberLoss(delta=1.0).to(device)

        
        for epoch in range(100):
            model.train()
            t_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                opt.step()
                t_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                v_loss = crit(model(val_x), val_y)
            # TensorBoard 记录
            writer.add_scalars(f'Fold_{fold+1}_Station_{val_station}', {'Train_Loss': t_loss/len(train_loader), 'Val_Loss': v_loss.item()}, epoch)
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_fold_{fold+1}.pth'))    
        # 记录 RMSE
        model.eval()
        with torch.no_grad():
            pred = scaler_y.inverse_transform(model(val_x).cpu().numpy())
            true = scaler_y.inverse_transform(val_y.cpu().numpy())
            rmse = np.sqrt(mean_squared_error(true, pred))
            writer.add_scalar('Final_RMSE_Station', rmse, fold+1)
    writer.close()
    print("所有站点验证完成，请在终端运行 tensorboard --logdir=logs 查看曲线")

# --- 7. 执行主流程 ---
if __name__ == "__main__":
    # 1. 地理基础
    grid_base = generate_base_grid()
    
    # 2. 静态 POI (半径 km)
    poi_static = aggregate_pois_radius(grid_base, FOLDER_POI, RADIUS_R)
    
    # 3. 空间映射
    grid_mapped, train_mapping = build_spatial_mapping(grid_base, PATH_METEO, PATH_AIR)
    
    # 4. 准备训练集
    train_df = prepare_final_dataset(train_mapping, grid_mapped, poi_static, PATH_METEO, PATH_AIR)
    
    # 5. 训练与监视
    train_loso(train_df)