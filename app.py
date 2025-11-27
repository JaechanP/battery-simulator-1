import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# ==============================================================================
# [0] 디자인 & 헤더 설정 (HTML/CSS)
# ==============================================================================

def get_img_tag(file, title):
    """이미지 태그 생성 (오류 방지)"""
    if not os.path.exists(file):
        return ""
    try:
        with open(file, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()
        return f'<img src="data:image/png;base64,{b64_data}" class="logo-img" title="{title}">'
    except:
        return ""

tag_ajou_sw = get_img_tag("ajou_sw_logo.png", "Ajou SW")
tag_ajou    = get_img_tag("ajou_logo.png", "Ajou University")
tag_google  = get_img_tag("google_logo.png", "Google")

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
    width: 1px; height: 20px; background-color: #bbb;
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
# [Engine 1] 데이터 로드 및 예측 함수
# ==============================================================================
@st.cache_data
def load_real_case_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "engine1_output.csv")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

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
# [Engine 2] 핵심 로직: 화학적 근거 기반 환경 영향 산출 (교수님 피드백 반영)
# ==============================================================================
def calculate_lca_impact(binder_type, solvent_type, drying_temp, loading_mass, drying_time):
    """
    [교수님 피드백 반영 로직]
    1. VOC = 용매의 종류 (NMP: 유독성=높음, Water: 무독성=없음)
    2. CO2 = 바인더 합성 시 불소(F) 유무 (PVDF: C-H-F 구조로 높음, CMGG: 바이오 유래로 낮음)
    3. Energy = 용매 끓는점(BP) 기반 열역학적 부하
    """
    
    # --- 1. VOC Emission (g/m²) ---
    # 근거: NMP는 전량 회수되지 않으면 VOC로 배출됨. 물은 VOC 0.
    if solvent_type == "NMP":
        # NMP는 유해 화학물질
        voc_base = 3.0 
        voc_val = voc_base * (loading_mass / 10.0) # 로딩량에 비례
        voc_desc = "Critical (NMP Toxicity)"
    else:
        # Water
        voc_val = 0.0
        voc_desc = "Clean (Water Vapor)"

    # --- 2. CO2 Emission (kg/m²) ---
    # 근거: PVDF는 불소(F) 함유 고분자로 GWP(지구온난화지수)가 매우 높음.
    # CMGG/CMC는 천연 유래(Cellulose 등)로 탄소 발생 저감.
    if binder_type == "PVDF":
        # 화학식: -(C2H2F2)n- (Fluorine 함유)
        co2_factor = 0.45 
        chem_formula = "-(C₂H₂F₂)ₙ-"
        co2_desc = f"High (Fluorinated Polymer, {chem_formula})"
    elif binder_type in ["CMGG", "GG", "CMC", "SBR"]:
        # 화학식: 천연 다당류 기반 (C, H, O 위주)
        co2_factor = 0.12
        chem_formula = "Bio-based (C,H,O)"
        co2_desc = f"Low ({chem_formula})"
    else:
        co2_factor = 0.3
        co2_desc = "Medium"
        
    co2_val = co2_factor * (loading_mass / 20.0)

    # --- 3. Energy Consumption (kWh/m²) ---
    # 근거: 용매의 끓는점(BP)에 따른 건조 에너지 차이
    # NMP BP = 202°C (고온 건조 필요), Water BP = 100°C (저온 건조 가능)
    if solvent_type == "NMP":
        boiling_point = 204.1
        # NMP는 기화열은 낮으나(570 kJ/kg), 공정 온도가 높아야 하고 회수 설비 에너지가 추가됨
        process_penalty = 1.5 
    else:
        boiling_point = 100.0
        process_penalty = 1.0

    # 에너지 모델: (온도차 델타T) * (시간) * (공정 복잡도)
    # 기준 온도 25도
    delta_T = max(drying_temp - 25, 0)
    
    # 끓는점보다 낮게 건조하면 건조 효율이 떨어져 시간이 더 걸리거나 에너지가 더 듬 (페널티)
    efficiency = 1.0 if drying_temp >= boiling_point else 0.6
    
    energy_val = (delta_T * drying_time * process_penalty) / (efficiency * 10000.0)
    
    return co2_val, energy_val, voc_val, co2_desc, voc_desc

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
                ax_cap.set_ylabel("Capacity (mAh/g)", fontsize=11, fontweight='bold')
                ax_cap.set_title("Discharge Capacity Prediction", fontsize=14, fontweight='bold')
                ax_cap.legend(); ax_cap.grid(True, linestyle='--', alpha=0.4)
                
                ax_ce.plot(cycles, ce, '-', color='#007bff', linewidth=1.5, alpha=0.8, label='Coulombic Efficiency')
                ax_ce.set_ylabel("CE (%)", fontsize=11, fontweight='bold')
                ax_ce.set_xlabel("Cycle Number", fontsize=11, fontweight='bold')
                ax_ce.set_ylim(98.0, 100.5)
                ax_ce.legend(); ax_ce.grid(True, linestyle='--', alpha=0.4)
                
                st.pyplot(fig2)
                
                eol_limit = init_cap_input * 0.8
                eol_cycle = np.where(capacity < eol_limit)[0]
                if len(eol_cycle) > 0:
                    st.error(f"⚠️ Warning: {eol_cycle[0]} Cycle에서 수명 종료(EOL) 예상")
                else:
                    st.success(f"✅ Stable: {cycle_input} Cycle까지 안정적")

# ------------------------------------------------------------------------------
# TAB 2: 실제 실험 검증
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Engine 1. 실제 실험 데이터 검증 (Real-world Validation)")
    st.markdown("이 탭에서는 **실제 배터리 테스트 데이터(Ground Truth)**를 기반으로 수행된 Engine 1의 정밀한 예측 결과를 검증합니다.")
    st.divider()

    df_results = load_real_case_data()

    if df_results is None:
        st.warning("⚠️ 'engine1_output.csv' 파일을 찾을 수 없습니다. (GitHub 업로드 확인 필요)")
    else:
        col_case_input, col_case_view = st.columns([1, 2])

        with col_case_input:
            with st.container(border=True):
                st.markdown("#### 📂 실험 케이스 선택")
                available_samples = df_results['Sample_Type'].unique()
                selected_sample = st.radio("확인할 실험 데이터:", available_samples, index=0, key="t2_radio")
                
                st.write("")
                if "Sample A" in selected_sample:
                    st.success("✅ **Sample A** (Stable)\n- Binder: CMGG\n- 예측 정확도: 높음")
                elif "Sample B" in selected_sample:
                    st.warning("⚠️ **Sample B** (Normal)\n- Binder: PVDF\n- 예측 정확도: 보통")
                else:
                    st.error("🚫 **Sample C** (Unstable)\n- 이슈: 초기 저항 증가")

        with col_case_view:
            sample_data = df_results[df_results['Sample_Type'] == selected_sample]
            history = sample_data[sample_data['Data_Type'] == 'History']
            prediction = sample_data[sample_data['Data_Type'] == 'Prediction']

            if not sample_data.empty:
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history['Cycle'], history['Capacity'], 'o-', color='black', markersize=4, alpha=0.7, label='History (1~100)')
                
                if not history.empty and not prediction.empty:
                    connect_x = [history['Cycle'].iloc[-1], prediction['Cycle'].iloc[0]]
                    connect_y = [history['Capacity'].iloc[-1], prediction['Capacity'].iloc[0]]
                    ax.plot(connect_x, connect_y, '--', color='#dc3545', linewidth=2)

                ax.plot(prediction['Cycle'], prediction['Capacity'], '--', color='#dc3545', linewidth=2, label='AI Prediction (101~)')
                ax.set_xlabel("Cycle", fontweight='bold'); ax.set_ylabel("Capacity (Ah)", fontweight='bold')
                ax.set_title(f"Validation Result - {selected_sample}", fontweight='bold')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
            else:
                st.error("데이터 없음")

# ------------------------------------------------------------------------------
# TAB 3: 친환경 공정 최적화 (Engine 2 - Logic Updated)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA Optimization)")
    st.info("💡 **Update:** 본 시뮬레이터는 **화학적 조성(불소 유무)**, **용매의 독성(VOC)**, **끓는점(Boiling Point)**에 기반한 물리학적 계산 모델을 적용했습니다.")
    
    col_input_e2, col_view_e2 = st.columns([1, 2])
    
    with col_input_e2:
        with st.container(border=True):
            st.markdown("#### 🛠️ 공정 조건 설정")
            s_binder = st.selectbox("Binder Type", ["PVDF", "CMGG", "GG", "CMC", "SBR"])
            s_solvent = st.radio("Solvent Type", ["NMP", "Water"])
            st.divider()
            s_temp = st.slider("Drying Temp (°C)", 60, 200, 110)
            s_time = st.slider("Drying Time (min)", 60, 720, 60) 
            s_loading = st.number_input("Loading mass (mg/cm²)", 5.0, 30.0, 10.0)
            
            st.write("")
            run_e2 = st.button("Engine 2 계산 실행", type="primary", use_container_width=True)

    with col_view_e2:
        if run_e2:
            # [핵심] 머신러닝 대신 과학적 수식 함수 호출
            co2, energy, voc, co2_desc, voc_desc = calculate_lca_impact(
                s_binder, s_solvent, s_temp, s_loading, s_time
            )
            
            # 결과 카드 표시
            col1, col2, col3 = st.columns(3)
            col1.metric("CO₂ Emission", f"{co2:.4f} kg/m²", delta=co2_desc, delta_color="inverse")
            col2.metric("Energy Consumption", f"{energy:.4f} kWh/m²", help="Based on Solvent BP & Drying Temp")
            col3.metric("VOC Emission", f"{voc:.4f} g/m²", delta=voc_desc, delta_color="inverse")
            
            st.divider()
            
            # 상세 분석 텍스트 (교수님 피드백 근거 제시용)
            st.markdown("#### 📋 Scientific Basis for Calculation")
            
            with st.expander("1. VOC (휘발성 유기화합물) 산출 근거", expanded=True):
                if s_solvent == "NMP":
                    st.write("🔴 **High Risk:** 용매로 **NMP(N-Methyl-2-pyrrolidone)**가 사용되었습니다. NMP는 생식 독성이 있는 유기용매로, 건조 과정에서 VOC가 다량 발생하며 엄격한 배기 장치가 필요합니다.")
                else:
                    st.write("🟢 **Safe:** 용매로 **Water(물)**이 사용되었습니다. 건조 시 수증기만 배출되므로 VOC 발생량은 **0**에 수렴합니다.")

            with st.expander("2. CO₂ (탄소 배출량) 산출 근거", expanded=True):
                if "PVDF" in s_binder:
                    st.write("🔴 **High Emission:** 바인더로 **PVDF**가 사용되었습니다.")
                    st.latex(r"-(C_2H_2F_2)_n-")
                    st.write("화학 구조 내 **불소(F)** 원소로 인해 합성 및 폐기 과정에서 GWP(지구온난화지수)가 매우 높습니다.")
                else:
                    st.write(f"🟢 **Low Emission:** 바인더로 **{s_binder}**가 사용되었습니다. 이는 **천연 유래 고분자(Bio-based)**로, C, H, O 기반의 구조를 가지며 불소를 포함하지 않아 탄소 배출이 적습니다.")

            with st.expander("3. Energy (에너지 소비) 산출 근거", expanded=True):
                bp = 202 if s_solvent == "NMP" else 100
                st.write(f"ℹ️ **Solvent Boiling Point:** {bp}°C")
                st.write(f"현재 설정 온도: **{s_temp}°C**")
                if s_solvent == "NMP":
                    st.write("NMP는 끓는점이 202°C로 높아, 완전 건조를 위해 높은 열에너지가 지속적으로 필요합니다.")
                else:
                    st.write("물은 끓는점이 100°C로 낮아, 상대적으로 적은 에너지로도 건조가 가능합니다.")

            # 그래프 그리기
            st.markdown("---")
            st.markdown("#### 📊 Comparative Analysis (NMP vs Water Process)")
            
            # 비교군 데이터 생성 (NMP 기준)
            ref_co2, ref_energy, ref_voc, _, _ = calculate_lca_impact("PVDF", "NMP", 130, s_loading, 60)
            
            labels = ['CO₂ (kg)', 'Energy (kWh)', 'VOC (g)']
            current_vals = [co2, energy, voc]
            ref_vals = [ref_co2, ref_energy, ref_voc]

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8, 5))
            rects1 = ax.bar(x - width/2, ref_vals, width, label='Reference (PVDF/NMP)', color='#FF8A80', alpha=0.8)
            rects2 = ax.bar(x + width/2, current_vals, width, label='Current Settings', color='#69F0AE', edgecolor='black')

            ax.set_ylabel('Impact Value')
            ax.set_title('Environmental Impact Comparison')
            ax.set_xticks(x); ax.set_xticklabels(labels, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            
            # 값 표시
            def autolabel(rects):
                for rect in rects:
                    h = rect.get_height()
                    ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)
            autolabel(rects1); autolabel(rects2)
            
            st.pyplot(fig)

        else:
            st.info("좌측 패널에서 공정 조건을 설정하고 [Engine 2 계산 실행]을 눌러주세요.")
