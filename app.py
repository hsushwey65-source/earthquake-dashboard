# =============================================================================
#  DOTEKHA — Earthquake Vulnerability ABM Dashboard
#  Run with: streamlit run app.py
#  Required files in the same folder: abm_data_full.csv, abm_data_final_step.csv
#  If CSVs are missing, the app regenerates them from the ABM automatically.
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOTEKHA · Earthquake Vulnerability Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global background & text ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] {
    background-color: #161b27 !important;
}
/* ---- metric cards ---- */
[data-testid="metric-container"] {
    background: #1c2333;
    border: 1px solid #2e3a50;
    border-radius: 10px;
    padding: 12px 18px;
}
/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #1c2333;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: #8892a4;
    border: 1px solid #2e3a50;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
}
/* ---- buttons ---- */
.stDownloadButton > button, .stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    background: #1d4ed8;
}
/* ---- dataframe ---- */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
/* ---- info / caption boxes ---- */
.info-box {
    background: #1c2333;
    border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin: 8px 0 16px 0;
    font-size: 0.88rem;
    color: #a0aec0;
}
/* ---- scenario colour pills ---- */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.pill-high   { background: #064e3b; color: #6ee7b7; }
.pill-middle { background: #1e3a5f; color: #93c5fd; }
.pill-low    { background: #4c1d1d; color: #fca5a5; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ABM CONSTANTS (mirror the notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS = np.array([
    [-0.70, -0.30, -0.10, -0.40, -0.20],
    [-0.30, -0.70, -0.10, -0.20, -0.10],
    [+0.10, +0.20, +0.60, +0.20, +0.30],
    [+0.20, +0.20, +0.80, +0.40, +0.40],
    [-0.20, -0.30, -0.30, -0.90, -0.20],
    [+0.50, +0.30, +0.20, +0.20, +0.80],
])
BIASES_H1   = np.array([0.10, 0.10, 0.05, 0.10, 0.05])
WEIGHTS_OUT = np.array([0.25, 0.25, 0.20, 0.20, 0.10])
BIAS_OUT    = 0.05

FEATURE_NAMES = ['income_level','education_level','age',
                 'health_limitation','transport_access','household_size']
HIDDEN_NAMES  = ['h1_economic','h2_social','h3_physical','h4_access','h5_household']

SCENARIO_LABELS = {1: 'High SES', 2: 'Middle SES', 3: 'Low SES'}
SCENARIO_COLORS = {'High SES': '#10b981', 'Middle SES': '#3b82f6', 'Low SES': '#f87171'}
COLOR_LIST      = ['#10b981', '#3b82f6', '#f87171']

# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION (fallback if CSVs are missing)
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_PARAMS = {
    1: {'income':(0.70,0.15),'education':(0.75,0.15),'age':(0.35,0.15),
        'health':(0.20,0.10),'transport':(0.80,0.15),'household':(0.25,0.10)},
    2: {'income':(0.50,0.15),'education':(0.50,0.15),'age':(0.50,0.20),
        'health':(0.45,0.15),'transport':(0.50,0.20),'household':(0.50,0.15)},
    3: {'income':(0.20,0.12),'education':(0.20,0.12),'age':(0.65,0.20),
        'health':(0.70,0.15),'transport':(0.15,0.10),'household':(0.75,0.15)},
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x_vec):
    h = sigmoid(x_vec @ WEIGHTS + BIASES_H1)
    v = float(sigmoid(h @ WEIGHTS_OUT + BIAS_OUT))
    return h, v

@st.cache_data(show_spinner="🔄 Generating simulation data…")
def generate_abm_data(n_agents=500, n_steps=50):
    np.random.seed(42)
    records = []
    for sc in [1, 2, 3]:
        label = SCENARIO_LABELS[sc]
        p = SCENARIO_PARAMS[sc]
        agents_x = np.column_stack([
            np.clip(np.random.normal(p['income'][0],    p['income'][1],    n_agents), 0, 1),
            np.clip(np.random.normal(p['education'][0], p['education'][1], n_agents), 0, 1),
            np.clip(np.random.normal(p['age'][0],       p['age'][1],       n_agents), 0, 1),
            np.clip(np.random.normal(p['health'][0],    p['health'][1],    n_agents), 0, 1),
            np.clip(np.random.normal(p['transport'][0], p['transport'][1], n_agents), 0, 1),
            np.clip(np.random.normal(p['household'][0], p['household'][1], n_agents), 0, 1),
        ])  # (n_agents, 6)
        for step in range(n_steps):
            if step > 0:
                agents_x = np.clip(agents_x + np.random.normal(0, 0.01, agents_x.shape), 0, 1)
            h_mat = sigmoid(agents_x @ WEIGHTS + BIASES_H1)          # (n_agents, 5)
            v_vec = sigmoid(h_mat @ WEIGHTS_OUT + BIAS_OUT)           # (n_agents,)
            for i in range(n_agents):
                rec = {
                    'scenario': sc, 'scenario_label': label,
                    'step': step, 'agent_id': i, 'vulnerability': float(v_vec[i]),
                }
                for j, fn in enumerate(FEATURE_NAMES): rec[fn] = float(agents_x[i, j])
                for j, hn in enumerate(HIDDEN_NAMES):  rec[hn] = float(h_mat[i, j])
                records.append(rec)
    return pd.DataFrame(records)

@st.cache_data(show_spinner="📂 Loading data…")
def load_data():
    try:
        df_full  = pd.read_csv('abm_data_full.csv')
        df_final = pd.read_csv('abm_data_final_step.csv')
    except FileNotFoundError:
        df_full  = generate_abm_data()
        max_step = df_full['step'].max()
        df_final = df_full[df_full['step'] == max_step].copy()
    return df_full, df_final

df_full, df_final = load_data()
ALL_STEPS   = sorted(df_full['step'].unique())
ALL_AGENTS  = sorted(df_full['agent_id'].unique())
MAX_STEP    = max(ALL_STEPS)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING (cached — runs once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="🤖 Training predictive models…")
def train_models(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    # XGBoost is optional — fails gracefully if libomp is missing on macOS
    try:
        import xgboost as xgb
        HAS_XGB = True
    except Exception:
        HAS_XGB = False

    X = df[FEATURE_NAMES].values
    y = df['vulnerability'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    results = {}

    # ── Random Forest ──
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    results['Random Forest'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_te, y_pred)), 6),
        'importance': rf.feature_importances_.tolist()
    }

    # ── XGBoost (skipped if library unavailable) ──
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                      verbosity=0, n_jobs=-1)
        xgb_model.fit(X_tr, y_tr)
        y_pred = xgb_model.predict(X_te)
        results['XGBoost'] = {
            'r2':   round(r2_score(y_te, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_te, y_pred)), 6),
            'importance': xgb_model.feature_importances_.tolist()
        }
    else:
        results['XGBoost'] = {
            'r2':   None, 'rmse': None, 'importance': None,
            'error': '⚠️ XGBoost unavailable — run `brew install libomp` to enable'
        }

    # ── Neural Network ──
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                      random_state=42, early_stopping=True)
    nn.fit(X_tr_sc, y_tr)
    y_pred = nn.predict(X_te_sc)
    results['Neural Network'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_te, y_pred)), 6),
        'importance': None
    }

    return results

model_results = train_models(df_final)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — GLOBAL CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://em-content.zobj.net/source/twitter/376/globe-showing-asia-australia_1f30f.png", width=48)
st.sidebar.title("DOTEKHA")
st.sidebar.caption("Earthquake Vulnerability ABM Dashboard")
st.sidebar.markdown("---")

# Scenario filter
scenario_options = ['All', 'High SES', 'Middle SES', 'Low SES']
selected_scenario = st.sidebar.selectbox(
    "🗂️ Scenario",
    scenario_options,
    help="Filter agents and charts by socioeconomic scenario."
)

# Time-step slider
selected_step = st.sidebar.slider(
    "⏱️ Time Step",
    min_value=int(min(ALL_STEPS)),
    max_value=int(MAX_STEP),
    value=int(MAX_STEP),
    step=1,
    help="Select which simulation time step to analyse."
)

# Derive filtered slice for the chosen step & scenario
df_step = df_full[df_full['step'] == selected_step].copy()
if selected_scenario != 'All':
    df_step_sc = df_step[df_step['scenario_label'] == selected_scenario]
else:
    df_step_sc = df_step

# Agent selector (dynamic)
agent_ids_available = sorted(df_step_sc['agent_id'].unique())
selected_agent = st.sidebar.selectbox(
    "🧑 Agent ID",
    agent_ids_available,
    help="Pick a specific agent to inspect in the Agent Explorer tab."
)

st.sidebar.markdown("---")
# Download button
csv_bytes = df_step_sc.to_csv(index=False).encode()
st.sidebar.download_button(
    "⬇️ Download Filtered Data",
    data=csv_bytes,
    file_name=f"abm_step{selected_step}_{selected_scenario.replace(' ','_')}.csv",
    mime="text/csv",
)
st.sidebar.markdown("---")
st.sidebar.caption("Built on Wisner / PAR Model · 500 agents × 50 steps × 3 scenarios")

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0;'>🌍 Earthquake Vulnerability Dashboard</h1>
<p style='color:#8892a4; margin-top:4px;'>
    Agent-Based Model · PAR / Wisner Framework · Interactive Research Tool
</p>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_sim, tab_hidden, tab_agent, tab_pred, tab_corr = st.tabs([
    "📊 Overview",
    "📈 Simulation",
    "🧠 Hidden Layer",
    "🔍 Agent Explorer",
    "🤖 Prediction",
    "🔗 Correlation",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader(f"Summary · Step {selected_step} · {selected_scenario}")

    # Top metrics
    m_mean = df_step_sc['vulnerability'].mean()
    m_min  = df_step_sc['vulnerability'].min()
    m_max  = df_step_sc['vulnerability'].max()
    m_n    = len(df_step_sc)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Vulnerability", f"{m_mean:.4f}")
    c2.metric("Min Vulnerability",  f"{m_min:.4f}")
    c3.metric("Max Vulnerability",  f"{m_max:.4f}")
    c4.metric("Number of Agents",   f"{m_n:,}")

    st.markdown("---")
    st.subheader("Scenario Comparison Cards")

    summary = df_step.groupby('scenario_label')['vulnerability'].agg(
        Mean='mean', Std='std', Min='min', Max='max', Median='median'
    ).round(4).reset_index()

    card_cols = st.columns(3)
    pill_cls  = {'High SES':'pill-high', 'Middle SES':'pill-middle', 'Low SES':'pill-low'}
    for i, row in summary.iterrows():
        col = card_cols[i % 3]
        pill = pill_cls.get(row['scenario_label'], '')
        col.markdown(f"""
        <div style='background:#1c2333; border:1px solid #2e3a50; border-radius:12px; padding:18px;'>
          <span class='pill {pill}'>{row['scenario_label']}</span>
          <h3 style='font-size:1.6rem; margin:10px 0 4px 0; color:#e0e0e0;'>{row['Mean']:.4f}</h3>
          <p style='color:#8892a4; font-size:0.82rem; margin:0;'>Mean Vulnerability</p>
          <hr style='border-color:#2e3a50; margin:10px 0;'>
          <table style='width:100%; font-size:0.82rem; color:#a0aec0;'>
            <tr><td>Median</td><td style='text-align:right;'>{row['Median']:.4f}</td></tr>
            <tr><td>Std Dev</td><td style='text-align:right;'>{row['Std']:.4f}</td></tr>
            <tr><td>Min</td><td style='text-align:right;'>{row['Min']:.4f}</td></tr>
            <tr><td>Max</td><td style='text-align:right;'>{row['Max']:.4f}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # Scenario mean vulnerability bar chart
    fig_bar = px.bar(
        summary, x='scenario_label', y='Mean', error_y='Std',
        color='scenario_label',
        color_discrete_map=SCENARIO_COLORS,
        labels={'scenario_label':'Scenario','Mean':'Mean Vulnerability'},
        title='Mean Vulnerability by Scenario (Current Step)',
        template='plotly_dark',
    )
    fig_bar.update_layout(showlegend=False, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("Simulation Analysis")
    st.markdown('<div class="info-box">📌 These charts show how vulnerability evolves over time and how agents distribute across the vulnerability spectrum.</div>', unsafe_allow_html=True)

    # Filter full dataset by scenario
    if selected_scenario != 'All':
        df_sim = df_full[df_full['scenario_label'] == selected_scenario]
    else:
        df_sim = df_full

    # ── Line chart: mean vulnerability over time ──
    ts = df_sim.groupby(['step','scenario_label'])['vulnerability'].mean().reset_index()
    fig_ts = px.line(
        ts, x='step', y='vulnerability', color='scenario_label',
        color_discrete_map=SCENARIO_COLORS,
        labels={'step':'Time Step','vulnerability':'Mean Vulnerability','scenario_label':'Scenario'},
        title='Mean Vulnerability Over Time (per Scenario)',
        template='plotly_dark',
        markers=False,
    )
    fig_ts.add_vline(x=selected_step, line_dash='dash', line_color='#facc15',
                     annotation_text=f"Step {selected_step}", annotation_position='top left')
    fig_ts.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    st.plotly_chart(fig_ts, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # ── Histogram: distribution at selected step ──
        fig_hist = px.histogram(
            df_step_sc, x='vulnerability', color='scenario_label',
            nbins=40, barmode='overlay', opacity=0.75,
            color_discrete_map=SCENARIO_COLORS,
            labels={'vulnerability':'Vulnerability Score','scenario_label':'Scenario'},
            title=f'Vulnerability Distribution · Step {selected_step}',
            template='plotly_dark',
        )
        fig_hist.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        # ── Boxplot: scenario comparison at selected step ──
        fig_box = px.box(
            df_step_sc, x='scenario_label', y='vulnerability',
            color='scenario_label', color_discrete_map=SCENARIO_COLORS,
            points='outliers',
            labels={'scenario_label':'Scenario','vulnerability':'Vulnerability Score'},
            title=f'Vulnerability Boxplot · Step {selected_step}',
            template='plotly_dark',
        )
        fig_box.update_layout(showlegend=False, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HIDDEN LAYER / THEORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_hidden:
    st.subheader("Hidden Layer — PAR / Wisner Dimensions")
    st.markdown("""
    <div class="info-box">
    🧠 <strong>Theory:</strong> This layer represents vulnerability dimensions derived from the
    <em>Pressure and Release (PAR) model</em> by Wisner et al.  Each hidden node aggregates
    input features through theory-driven weights — capturing economic, social, physical,
    access, and household dimensions of vulnerability.
    </div>
    """, unsafe_allow_html=True)

    # Mean hidden values per scenario at selected step
    h_data = df_step.groupby('scenario_label')[HIDDEN_NAMES].mean().reset_index()
    h_melted = h_data.melt(id_vars='scenario_label', var_name='Hidden Node', value_name='Mean Activation')
    h_melted['Hidden Node'] = h_melted['Hidden Node'].map({
        'h1_economic':  'H1 · Economic',
        'h2_social':    'H2 · Social',
        'h3_physical':  'H3 · Physical',
        'h4_access':    'H4 · Access',
        'h5_household': 'H5 · Household',
    })

    fig_hbar = px.bar(
        h_melted, x='Hidden Node', y='Mean Activation',
        color='scenario_label', barmode='group',
        color_discrete_map=SCENARIO_COLORS,
        labels={'scenario_label':'Scenario'},
        title=f'Mean Hidden Node Activations by Scenario · Step {selected_step}',
        template='plotly_dark',
    )
    fig_hbar.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                           xaxis_tickangle=-20)
    st.plotly_chart(fig_hbar, use_container_width=True)

    # Radar chart per scenario
    st.subheader("Radar Chart — Vulnerability Dimensions")
    radar_cols = st.columns(3)
    scenarios_list = ['High SES', 'Middle SES', 'Low SES']
    for i, sc in enumerate(scenarios_list):
        sc_row = h_data[h_data['scenario_label'] == sc]
        if sc_row.empty:
            continue
        vals = sc_row[HIDDEN_NAMES].values.flatten().tolist()
        labels_radar = ['H1 Economic','H2 Social','H3 Physical','H4 Access','H5 Household']
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels_radar + [labels_radar[0]],
            fill='toself',
            name=sc,
            line_color=SCENARIO_COLORS[sc],
            fillcolor=SCENARIO_COLORS[sc],
            opacity=0.35,
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#1c2333',
                radialaxis=dict(visible=True, range=[0, 0.8], color='#8892a4'),
                angularaxis=dict(color='#8892a4'),
            ),
            showlegend=False,
            title=dict(text=sc, font_color=SCENARIO_COLORS[sc]),
            paper_bgcolor='#0e1117',
            margin=dict(t=50, b=30, l=30, r=30),
        )
        radar_cols[i].plotly_chart(fig_radar, use_container_width=True)

    # Data table
    with st.expander("📋 Show Hidden Node Mean Values Table"):
        st.dataframe(h_data.style.background_gradient(cmap='Blues', subset=HIDDEN_NAMES), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AGENT EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_agent:
    st.subheader(f"Agent Explorer · Agent {selected_agent}")
    st.markdown('<div class="info-box">🔍 Inspect an individual agent\'s inputs, hidden vulnerability dimensions, and final vulnerability score.</div>', unsafe_allow_html=True)

    # Fetch agent row at selected step
    agent_row = df_full[
        (df_full['agent_id'] == selected_agent) &
        (df_full['step'] == selected_step)
    ]
    if selected_scenario != 'All':
        agent_row = agent_row[agent_row['scenario_label'] == selected_scenario]

    if agent_row.empty:
        st.warning("No data found for this agent/step/scenario combination. Adjust the sidebar filters.")
    else:
        row = agent_row.iloc[0]

        # Top cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Scenario",        row['scenario_label'])
        c2.metric("Time Step",       int(row['step']))
        c3.metric("Vulnerability",   f"{row['vulnerability']:.4f}")

        st.markdown("---")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 📥 Input Variables")
            input_df = pd.DataFrame({
                'Feature': ['Income Level','Education Level','Age',
                            'Health Limitation','Transport Access','Household Size'],
                'Value':   [round(row[f], 4) for f in FEATURE_NAMES]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)

            st.markdown("#### 🧠 Hidden Layer Values")
            hidden_df = pd.DataFrame({
                'Dimension':   ['H1 Economic','H2 Social','H3 Physical','H4 Access','H5 Household'],
                'Activation':  [round(row[h], 4) for h in HIDDEN_NAMES]
            })
            st.dataframe(hidden_df, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### 🕸️ Agent Radar Chart")
            # Combine inputs + hidden for radar
            radar_vals   = [row[f] for f in FEATURE_NAMES] + [row[h] for h in HIDDEN_NAMES]
            radar_labels = ['Income','Education','Age','Health','Transport','Household',
                            'H1 Econ','H2 Social','H3 Phys','H4 Access','H5 Hhold']
            sc_color = SCENARIO_COLORS.get(row['scenario_label'], '#3b82f6')
            fig_agent = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                line_color=sc_color,
                fillcolor=sc_color,
                opacity=0.4,
                name=f"Agent {selected_agent}",
            ))
            fig_agent.update_layout(
                polar=dict(
                    bgcolor='#1c2333',
                    radialaxis=dict(visible=True, range=[0, 1], color='#8892a4'),
                    angularaxis=dict(color='#8892a4'),
                ),
                paper_bgcolor='#0e1117',
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=400,
            )
            st.plotly_chart(fig_agent, use_container_width=True)

        # Evolution chart for this agent over time
        st.markdown("---")
        st.markdown("#### 📈 Agent Vulnerability Over Time")
        agent_ts = df_full[df_full['agent_id'] == selected_agent].copy()
        if selected_scenario != 'All':
            agent_ts = agent_ts[agent_ts['scenario_label'] == selected_scenario]
        fig_ats = px.line(
            agent_ts.sort_values('step'), x='step', y='vulnerability',
            color='scenario_label', color_discrete_map=SCENARIO_COLORS,
            labels={'step':'Time Step','vulnerability':'Vulnerability','scenario_label':'Scenario'},
            title=f'Vulnerability Trajectory · Agent {selected_agent}',
            template='plotly_dark',
        )
        fig_ats.add_vline(x=selected_step, line_dash='dash', line_color='#facc15')
        fig_ats.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        st.plotly_chart(fig_ats, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PREDICTIVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.subheader("Predictive Model Performance")
    st.markdown('<div class="info-box">🤖 Three models (Neural Network, Random Forest, XGBoost) were trained on the final-step snapshot to predict agent vulnerability from raw input features.</div>', unsafe_allow_html=True)

    # Metrics table
    metrics_df = pd.DataFrame([
        {'Model': m,
         'R²':   v['r2']   if v['r2']   is not None else '—',
         'RMSE': v['rmse'] if v['rmse'] is not None else '—'}
        for m, v in model_results.items()
    ])

    col_m, col_chart = st.columns([1, 2])
    with col_m:
        st.markdown("#### 📋 Metrics Summary")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col_chart:
        fig_r2 = px.bar(
            metrics_df, x='Model', y='R²',
            color='Model',
            color_discrete_sequence=['#10b981','#3b82f6','#f87171'],
            title='R² Score Comparison',
            template='plotly_dark',
            text='R²',
        )
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(showlegend=False, paper_bgcolor='#0e1117',
                             plot_bgcolor='#0e1117', yaxis_range=[0, 1.05])
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Importance (Random Forest & XGBoost)")
    feat_col1, feat_col2 = st.columns(2)

    for col, model_name in zip([feat_col1, feat_col2], ['Random Forest', 'XGBoost']):
        imp = model_results[model_name].get('importance')
        if imp is None:
            err = model_results[model_name].get('error', f'{model_name} feature importance unavailable.')
            col.warning(err)
            continue
        imp_df = pd.DataFrame({
            'Feature':    ['Income Level','Education Level','Age',
                           'Health Limitation','Transport Access','Household Size'],
            'Importance': imp
        }).sort_values('Importance', ascending=True)
        fig_fi = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            title=f'{model_name} · Feature Importance',
            template='plotly_dark',
            color='Importance',
            color_continuous_scale='Blues',
        )
        fig_fi.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                             coloraxis_showscale=False)
        col.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.subheader("Correlation Analysis")
    st.markdown('<div class="info-box">🔗 Explore linear correlations between input features, hidden vulnerability dimensions, and the output vulnerability score.</div>', unsafe_allow_html=True)

    # Toggle: full vs final step
    corr_scope = st.radio(
        "Dataset scope",
        ["Final Step Only", "Full Dataset (all time steps)"],
        horizontal=True
    )
    df_corr_base = df_final if corr_scope == "Final Step Only" else df_full

    # Scenario filter for correlation
    if selected_scenario != 'All':
        df_corr_base = df_corr_base[df_corr_base['scenario_label'] == selected_scenario]

    corr_cols = FEATURE_NAMES + HIDDEN_NAMES + ['vulnerability']
    corr_labels = ['Income','Education','Age','Health','Transport','Household',
                   'H1 Econ','H2 Social','H3 Phys','H4 Access','H5 Hhold','Vulnerability']
    corr_mat = df_corr_base[corr_cols].corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_labels,
        y=corr_labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr_mat.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hoverongaps=False,
        colorbar=dict(title='r'),
    ))
    fig_heat.update_layout(
        title=f'Correlation Matrix · {selected_scenario} · {corr_scope}',
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        height=560,
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Top correlations with vulnerability
    st.markdown("#### Top Correlations with Vulnerability Score")
    vuln_corr = corr_mat['vulnerability'].drop('vulnerability').abs().sort_values(ascending=False)
    fig_vc = px.bar(
        x=vuln_corr.index.map(dict(zip(corr_cols[:-1], corr_labels[:-1]))),
        y=vuln_corr.values,
        labels={'x':'Variable','y':'|Correlation| with Vulnerability'},
        title='Absolute Correlation with Vulnerability',
        template='plotly_dark',
        color=vuln_corr.values,
        color_continuous_scale='Blues',
    )
    fig_vc.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                         coloraxis_showscale=False)
    st.plotly_chart(fig_vc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>
  DOTEKHA · Agent-Based Earthquake Vulnerability Model ·
  Wisner / PAR Framework · Built with Streamlit + Plotly
</p>
""", unsafe_allow_html=True)
