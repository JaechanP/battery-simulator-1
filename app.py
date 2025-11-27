import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# ==============================================================================
# [0] 디자인 & 헤더 설정 (HTML/CSS)
# ==============================================================================

def get_img_tag(file, title):
    """
    이미지 파일을 읽어서 완벽한 HTML <img> 태그를 반환하는 함수
    (파일이 없거나 에러 발생 시 빈 문자열 반환하여 화면 깨짐 방지)
    """
    if not os.path.exists(file):
        return ""
    try:
        with open(file, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()
        return f'<img src="data:image/png;base64,{b64_data}" class="logo-img" title="{title}">'
    except:
        return ""

# 로고 태그 생성
tag_ajou_sw = get_img_tag("ajou_sw_logo.png", "Ajou SW")
tag_ajou    = get_img_tag("ajou_logo.png", "Ajou University")
tag_google  = get_img_tag("google_logo.png", "Google")

# HTML/CSS 스타일링
header_html = f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Helvetica Neue', 'Apple SD Gothic Neo', sans-serif;
}}
.header-container {{
    background-color: #E8F5E9;
    padding: 30px 20px;
    border-radius: 20px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-bottom: 5px solid #4CAF50;
}}
.main-title {{
    font-size: 2.5rem;
    font-weight: 900;
    color: #1B5E20;
    margin: 0;
    padding-bottom: 5px;
    white-space: nowrap;
    letter-spacing: -1px;
}}
.sub-title {{
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 20px;
    font-weight: 500;
}}
.logo-box {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin-top: 5px;
}}
.logo-img {{
    height: 30px;
    width: auto;
    object-fit: contain;
    transition: transform 0.3s;
}}
.logo-img:hover {{
    transform: scale(1.1);
}}
.separator {{
    width: 1px; 
    height: 20px; 
    background-color: #bbb;
}}
@media (max-width: 900px) {{
    .main-title {{ font-size: 1.8rem; white-space: normal; }}
    .logo-img {{ height: 25px; }}
}}
</style>

<div class="header-container">
    <h1 class="main-title">AI 기반 배터리 소재/공정 최적화 시뮬레이터</h1>
    <div class="sub-title">Team 스물다섯 | Google-아주대학교 AI 융합 캡스톤 디자인</div>
    <div class="logo-box">
        {tag_ajou_sw}
        {tag_ajou}
        <div class="separator"></div>
        {tag_google}
    </div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)

st.info("""이 플랫폼은 Engine 1(수명 예측)과 Engine 2(환경 영향 평가)를 통합한 시뮬레이터입니다. 아래 탭을 선택하여 기능을 사용해보세요.""")

# ==============================================================================
# [데이터 로드 함수 모음]
# ==============================================================================

# 1. Engine 2 모델 로드 (업데이트된 로직 적용)
@st.cache_resource
def load_engine2_model():
    try:
        # 실제 엑셀 파일 로드 시도
        db = pd.read_excel('engine2_database.xlsx', sheet_name='LCA_Data', engine='openpyxl')
    except:
        # ------------------------------------------------------------------
        # [수정됨] 슬라이더 입력 범위(Time 720, Temp 180)를 모두 커버하도록 학습 데이터 확장
        # ------------------------------------------------------------------
        n = 1000 # 데이터 개수 증가 (학습 안정성 확보)
        
        # (1) 입력 변수 생성
        binder_types = np.random.choice(['PVDF', 'CMGG', 'GG', 'CMC'], n)
        solvent_types = []
        
        # PVDF는 주로 NMP와 짝을 이룸
        for b in binder_types:
            if b == 'PVDF': 
                solvent_types.append('NMP')
            else: 
                solvent_types.append(np.random.choice(['Water', 'Water', 'NMP']))

        # [중요] 학습 데이터의 범위를 슬라이더 최대치보다 조금 더 넓게 설정
        temp = np.random.uniform(60, 190, n)       # 온도 (60~180 커버)
        time = np.random.uniform(10, 750, n)       # 시간 (10~720 커버 - 기존 120에서 대폭 수정)
        loading = np.random.uniform(1.0, 55.0, n)  # 로딩양 (1~50 커버)
        
        # (2) 출력 변수 생성 (공학적 상관관계 적용)
        energy_list = []
        co2_list = []
        voc_list = []
        
        for i in range(n):
            # 기본 에너지 소비량 계산 (단위: kWh/m2)
            # 계수 조정: 범위가 커졌으므로 값이 너무 튀지 않게 계수를 미세 조정함
            # - Temp: 높을수록 에너지 증가
            # - Time: 길수록 에너지 비례 증가 (가장 큰 영향)
            # - Loading: 많을수록 증발 잠열 증가
            base_energy = (temp[i] * 0.002) + (time[i] * 0.0005) + (loading[i] * 0.002)
            
            if solvent_types[i] == 'NMP':
                # NMP는 회수 장치 부하로 에너지/CO2/VOC 높음
                e_val = base_energy + 0.4 
                c_val = e_val * 0.4 + 0.1 
                v_val = 3.0 + np.random.normal(0, 0.2)
            else: # Water
                # Water는 친환경
                e_val = base_energy 
                c_val = e_val * 0.2 + 0.05
                v_val = 0.0
            
            energy_list.append(e_val)
            co2_list.append(c_val)
            voc_list.append(v_val)

        # 데이터프레임 생성
        data = {
            'Binder_Type': binder_types,
            'Solvent_Type': solvent_types,
            'Binder_Amount_wt': np.random.uniform(1, 5, n),
            'Graphite_wt': np.random.uniform(90, 98, n),
            'SuperP_wt': np.random.uniform(0.5, 2, n),
            'Coating_Thickness_mm': np.random.uniform(0.05, 0.2, n),
            'Drying_Temp_C': temp,
            'Drying_Time_min': time,
            'Areal_Mass_Loading_mg_cm^2': loading, # [수정] UI 입력 단위(mg/cm^2)와 컬럼명 통일
            'CO2_kg_per_m2': co2_list,
            'Energy_kWh_per_m2': energy_list,
            'VOC_g_per_m2': voc_list
        }
        db = pd.DataFrame(data)

    X = db.drop(columns=['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'], errors='ignore')
    targets = [c for c in ['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'] if c in db.columns]
    Y = db[targets]
    
    numeric_features = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
    categorical_features = [c for c in X.columns if X[c].dtype == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_processed, Y)
    
    return model, preprocessor, db

# 2. Engine 1 실제 검증 데이터 로드 (CSV)
@st.cache_data
def load_real_case_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "engine1_output.csv")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

# ==============================================================================
# [Engine 1] 가상 시뮬레이터용 수명 예측 함수
# ==============================================================================
def predict_life_and_ce(decay_rate, specific_cap_base=185.0, cycles=1000):
    x = np.arange(1, cycles + 1)
    
    linear_fade = 0.00015 * x * decay_rate
    acc_fade = 1e-9 * np.exp(0.015 * x) * decay_rate
    cap_noise = np.random.normal(0, 0.0015, size=len(x))
    
    retention = 1.0 - linear_fade - acc_fade + cap_noise
    capacity = retention * specific_cap_base
    
    if decay_rate < 2.0:
        base_ce = 99.95; ce_noise_scale = 0.02
    elif decay_rate < 4.0:
        base_ce = 99.85; ce_noise_scale = 0.05
    else:
        base_ce = 99.6 - (x * 0.0008); ce_noise_scale = 0.15
        
    ce_noise = np.random.normal(0, ce_noise_scale, size=len(x))
    ce = np.clip(base_ce + ce_noise, 0, 100.0)

    return x, np.clip(capacity, 0, None), ce

# ==============================================================================
# [메인 UI] 탭 구성
# ==============================================================================

tab1, tab2, tab3 = st.tabs([
    "🧪 Engine 1-1: 가상 시뮬레이터", 
    "📊 Engine 1-2: 실제 실험 검증", 
    "🏭 Engine 2: 친환경 공정 최적화"
])

# ------------------------------------------------------------------------------
# TAB 1: 가상 시뮬레이터
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Engine 1. 배터리 수명 가상 시뮬레이터 (Interactive Mode)")
    st.markdown("사용자가 **직접 변수(초기 용량, 목표 사이클)를 조절**하며 AI 모델의 예측 경향성을 빠르게 파악하는 교육용 시뮬레이터입니다.")
    st.divider()
    
    col_input, col_view = st.columns([1, 2])
    with col_input:
        with st.container(border=True):
            st.markdown("#### 🧪 테스트 샘플 선택")
            sample_type = st.radio(
                "패턴 선택",
                ["Sample A (안정적 - CMGG)", "Sample B (일반적 - PVDF)", "Sample C (불안정 - 초기불량)"],
                label_visibility="collapsed",
                key="t1_radio"
            )
            st.divider()
            st.markdown("#### ⚙️ 예측 조건 설정")
            init_cap_input = st.number_input("Initial specific capacity (mAh/g)", 100.0, 400.0, 185.0)
            cycle_input = st.number_input("Number of cycles for prediction", 200, 5000, 1000, step=100)
            
            run_e1 = st.button("가상 예측 실행", type="primary", use_container_width=True)

    with col_view:
        if run_e1:
            with st.spinner("AI Analyzing..."):
                if "Sample A" in sample_type:
                    decay = 1.0; label = "Excellent (CMGG)"; color = '#28a745'
                elif "Sample B" in sample_type:
                    decay = 2.5; label = "Normal (PVDF)"; color = '#fd7e14'
                else:
                    decay = 5.0; label = "Poor (Defective)"; color = '#dc3545'
                
                cycles, capacity, ce = predict_life_and_ce(decay_rate=decay, specific_cap_base=init_cap_input, cycles=cycle_input)
                
                plt.style.use('default')
                fig2, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                ax_cap.plot(cycles[:100], capacity[:100], 'k-', linewidth=2.5, label='Input Data (1~100)')
                ax_cap.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2.5, label=f'AI Prediction ({label})')
                ax_cap.set_ylabel("Specific Capacity (mAh/g)", fontsize=11, fontweight='bold')
                ax_cap.set_title("Discharge Capacity Prediction", fontsize=14, fontweight='bold', pad=15)
                ax_cap.legend(loc='upper right', frameon=True, shadow=True)
                ax_cap.grid(True, linestyle='--', alpha=0.4)
                ax_cap.spines['top'].set_visible(False); ax_cap.spines['right'].set_visible(False)
                
                ax_ce.plot(cycles, ce, '-', color='#007bff', linewidth=1.5, alpha=0.8, label='Coulombic Efficiency')
                ax_ce.set_ylabel("Coulombic Efficiency (%)", fontsize=11, fontweight='bold')
                ax_ce.set_xlabel("Cycle Number", fontsize=11, fontweight='bold')
                ax_ce.set_ylim(98.0, 100.5)
                ax_ce.legend(loc='lower right', frameon=True, shadow=True)
                ax_ce.grid(True, linestyle='--', alpha=0.4)
                ax_ce.spines['top'].set_visible(False); ax_ce.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                eol_limit = init_cap_input * 0.8
                eol_cycle = np.where(capacity < eol_limit)[0]
                
                st.markdown("#### 📊 AI Analysis Report")
                if len(eol_cycle) > 0:
                    st.error(f"⚠️ **Warning:** 약 **{eol_cycle[0]} Cycle**에서 수명이 80%({eol_limit:.1f} mAh/g) 이하로 떨어질 것으로 예상됩니다.")
                else:
                    st.success(f"✅ **Stable:** 설정한 **{cycle_input} Cycle**까지 수명이 80% 이상 안정적으로 유지됩니다.")
        else:
            st.info("좌측 패널에서 조건을 설정하고 [가상 예측 실행]을 눌러주세요.")

# ------------------------------------------------------------------------------
# TAB 2: 실제 실험 검증
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Engine 1. 실제 실험 데이터 검증 (Real-world Validation)")
    st.markdown("""
    이 탭에서는 실제 배터리 테스트 데이터(Ground Truth)를 기반으로 수행된 Engine 1의 정밀한 예측 결과를 검증합니다.
    (Engine 1 ML 모델을 통해 사전 연산된 데이터를 로드합니다.)
    """)
    st.divider()

    df_results = load_real_case_data()

    if df_results is None:
        st.warning("⚠️ 'engine1_output.csv' 파일을 찾을 수 없습니다. 로컬에서 main_engine1.py를 실행하여 결과 파일을 생성해주세요.")
    else:
        col_case_input, col_case_view = st.columns([1, 2])

        with col_case_input:
            with st.container(border=True):
                st.markdown("#### 📂 실험 케이스 선택")
                available_samples = df_results['Sample_Type'].unique()
                selected_sample = st.radio(
                    "확인할 실험 데이터를 선택하세요:",
                    available_samples,
                    index=0,
                    key="t2_radio"
                )
                st.write("")
                if "Sample A" in selected_sample:
                    st.success("✅ **Sample A**\n\n- 상태: 안정적 (Stable)\n- Binder: CMGG\n- 예측 정확도: 높음")
                elif "Sample B" in selected_sample:
                    st.warning("⚠️ **Sample B**\n\n- 상태: 일반적 (Normal)\n- Binder: PVDF\n- 예측 정확도: 보통")
                else:
                    st.error("🚫 **Sample C**\n\n- 상태: 불안정 (Unstable)\n- 이슈: 초기 저항 증가")

        with col_case_view:
            sample_data = df_results[df_results['Sample_Type'] == selected_sample]
            history = sample_data[sample_data['Data_Type'] == 'History']
            prediction = sample_data[sample_data['Data_Type'] == 'Prediction']

            if not sample_data.empty:
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(history['Cycle'], history['Capacity'], 'o-', color='black', markersize=4, alpha=0.7, label='Input History (Cycle 1~100)')

                if not history.empty and not prediction.empty:
                    connect_x = [history['Cycle'].iloc[-1], prediction['Cycle'].iloc[0]]
                    connect_y = [history['Capacity'].iloc[-1], prediction['Capacity'].iloc[0]]
                    ax.plot(connect_x, connect_y, '--', color='#dc3545', linewidth=2)

                ax.plot(prediction['Cycle'], prediction['Capacity'], '--', color='#dc3545', linewidth=2, label='AI Prediction (Cycle 101~)')

                ax.set_xlabel("Cycle Number", fontsize=12, fontweight='bold')
                ax.set_ylabel("Discharge Capacity (Ah)", fontsize=12, fontweight='bold')
                ax.set_title(f"Model Validation Result - {selected_sample}", fontsize=14, fontweight='bold', pad=15)
                ax.legend(fontsize=11)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

                st.pyplot(fig)
                
                if not prediction.empty:
                    final_cycle = prediction['Cycle'].iloc[-1]
                    final_cap = prediction['Capacity'].iloc[-1]
                    st.info(f"📊 **AI 분석 리포트**: {selected_sample}은 **{int(final_cycle)} Cycle**까지 예측되었으며, 최종 용량은 **{final_cap:.3f} Ah**로 예상됩니다.")
            else:
                st.error("선택한 샘플의 데이터가 비어있습니다.")

# ------------------------------------------------------------------------------
# TAB 3: 친환경 공정 최적화
# ------------------------------------------------------------------------------
with tab3:
    model_e2, prep_e2, db_e2 = load_engine2_model()
    
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA Optimization)")
    
    col_input_e2, col_view_e2 = st.columns([1, 2])
    
    with col_input_e2:
        with st.container(border=True):
            st.markdown("#### 🛠️ 공정 조건 설정")
            s_binder = st.selectbox("Binder Type", ["PVDF", "CMGG", "GG", "CMC"])
            s_solvent = st.radio("Solvent Type", ["NMP", "Water"])
            st.divider()
            s_temp = st.slider("Drying Temp (°C)", 60, 180, 110)
            s_time = st.slider("Drying Time (min)", 10, 720, 120) 
            s_loading = st.number_input("Loading mass (mg/cm²)", 1.0, 50.0, 20.0)
            st.write("")
            run_e2 = st.button("Engine 2 예측 실행", type="primary", use_container_width=True)

    with col_view_e2:
        if run_e2:
            input_data = pd.DataFrame({
                'Binder_Type': [s_binder], 'Solvent_Type': [s_solvent],
                'Binder_Amount_wt': [2.0], 'Graphite_wt': [97.0], 'SuperP_wt': [1.0],
                'Coating_Thickness_mm': [0.1], 
                'Drying_Temp_C': [s_temp], 'Drying_Time_min': [s_time],
                'Areal_Mass_Loading_mg_cm^2': [s_loading]
            })
            
            try:
                X_new = prep_e2.transform(input_data)
                pred = model_e2.predict(X_new)[0] 
                
                col1, col2, col3 = st.columns(3)
                col1.metric("CO₂ Emission", f"{pred[0]:.4f} kg/m²", delta="Low Carbon" if pred[0] < 0.1 else "High Carbon", delta_color="inverse")
                col2.metric("Energy Consumption", f"{pred[1]:.4f} kWh/m²")
                col3.metric("VOC Emission", f"{pred[2]:.4f} g/m²", delta="-100%" if pred[2]<0.01 else None, delta_color="inverse")
                
                st.divider()
                st.markdown("#### 📊 Environmental Impact Comparison")
                nmp_mean = db_e2[db_e2['Solvent_Type']=='NMP'][['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2']].mean()
                if nmp_mean.isnull().all():
                    nmp_mean = pd.Series([0.27, 0.6, 3.0], index=['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'])

                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(3)
                width = 0.35
                
                color_nmp = '#FF8A80'
                color_sim = '#69F0AE'
                
                rects1 = ax.bar(x - width/2, nmp_mean.values, width, label='Reference (NMP)', color=color_nmp, edgecolor='white', alpha=0.9)
                rects2 = ax.bar(x + width/2, pred, width, label='Current Simulation', color=color_sim, edgecolor='gray', linewidth=1)
                
                ax.set_xticks(x)
                ax.set_xticklabels(['CO2', 'Energy', 'VOC'], fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=11)
                ax.set_title('Environmental Impact Comparison', fontsize=14, fontweight='bold', pad=15)
                ax.legend(fontsize=10, frameon=True, shadow=True)
                ax.grid(axis='y', linestyle=':', alpha=0.6)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                
                autolabel(rects1)
                autolabel(rects2)
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"예측 오류: {e}")
        else:
            st.info("좌측 패널에서 공정 조건을 설정하고 [Engine 2 예측 실행]을 눌러주세요.")
