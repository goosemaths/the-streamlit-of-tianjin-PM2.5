import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import duckdb

# --- 页面配置 ---
st.set_page_config(page_title="天津空气污染 3D 重构", layout="wide")

# --- 1. 数据高效读取与去重 (核心：解决网格数翻倍问题) ---
@st.cache_resource
def get_db_connection():
    return duckdb.connect(database=':memory:')

db = get_db_connection()

@st.cache_data
def get_all_timestamps():
    res = db.execute("SELECT DISTINCT dt FROM 'tianjin_pm25_predictions.parquet' ORDER BY dt").df()
    return res['dt'].tolist()

@st.cache_data
def get_static_data():
    # 读取静态表并强制去重，防止网格叠加
    df = pd.read_csv("grid_static_attributes.csv")
    return df.drop_duplicates(subset=['grid_id'])

def get_data_for_hour(selected_time):
    # 读取当前小时数据并强制去重
    query = f"SELECT id, v FROM 'tianjin_pm25_predictions.parquet' WHERE dt = '{selected_time}'"
    df = db.execute(query).df()
    return df.drop_duplicates(subset=['id'])

# --- 页面逻辑 ---
st.title("🏙️ 天津市空气污染 3D 时空重构")

try:
    all_times = get_all_timestamps()
    df_static = get_static_data()

    # 侧边栏
    selected_time = st.sidebar.select_slider(
        "选择模拟时刻",
        options=all_times,
        format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:00')
    )

    # 获取并合并数据
    current_hour_data = get_data_for_hour(selected_time)
    # 合并 (使用 inner join 确保只保留匹配的网格)
    plot_df = pd.merge(current_hour_data, df_static, left_on='id', right_on='grid_id', how='inner')

    # 处理异常值
    plot_df['v'] = plot_df['v'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 500)

    # --- 指标看板 (这里现在应该显示 2.2万左右) ---
    c1, c2, c3 = st.columns(3)
    c1.metric("全市均值", f"{plot_df['v'].mean():.1f} μg/m³")
    c2.metric("最高值", f"{plot_df['v'].max():.1f} μg/m³")
    c3.metric("网格总数", f"{len(plot_df)}")

    # 颜色逻辑 (保持代码1的配色逻辑)
    def get_color(v):
        if v < 35: return [0, 228, 0, 90]
        if v < 75: return [255, 255, 0, 100]
        if v < 115: return [255, 126, 0, 120]
        return [255, 0, 0, 130]
    
    plot_df['color'] = plot_df['v'].apply(get_color)
    
    DARK_LABELS_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    # --- 2. 地图渲染与悬停 (集成代码2的 POI 信息) ---
    layer = pdk.Layer(
        "ColumnLayer",
        plot_df,
        get_position=["lng_wgs84", "lat_wgs84"],
        get_elevation="v",
        elevation_scale=60,      # 略微调高，增加 3D 感
        radius=400,              # 减小半径到 400，防止网格完全盖住底图行政信息
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # 悬停内容：请确保字段名（如 cnt_industrial）与你的 csv 列名完全一致
    tooltip_content = {
        "html": """
            <div style="font-family: sans-serif; padding: 10px; background-color: rgba(0,0,0,0.7); border-radius: 5px;">
                <b style="font-size: 14px;">网格 ID: {grid_id}</b><br/>
                <b style="color: #FFD700;">浓度: {v} μg/m³</b><br/>
                <hr style="margin: 5px 0; border-top: 1px solid #eee;">
                <div style="font-size: 12px; line-height: 1.6;">
                    🏭 工厂企业: {cnt_industrial}<br/>
                    🛍️ 商业文化: {cnt_commercial}<br/>
                    🌳 风景自然: {cnt_nature}<br/>
                    🚌 交通设施: {cnt_transport}<br/>
                    🍴 餐饮服务: {cnt_catering}
                </div>
            </div>
        """,
        "style": {"color": "white", "border": "none", "zIndex": 1000}
    }

    # 视图设置
    view_state = pdk.ViewState(
        longitude=117.3, 
        latitude=39.1, 
        zoom=11.5, 
        pitch=45, 
        bearing=0
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        # 使用 dark-v10 保证能看到行政边界和标注
        map_style=DARK_LABELS_STYLE, 
        tooltip=tooltip_content
    ))

except Exception as e:
    st.error(f"渲染失败: {e}")
    st.info("提示：请检查 grid_static_attributes.csv 中的 POI 字段名是否为 cnt_industrial, cnt_commercial 等。")