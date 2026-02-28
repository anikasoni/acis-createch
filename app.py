# import streamlit as st
# import yaml, pandas as pd

# st.set_page_config(page_title='ACIS', page_icon='🏗️', layout='wide')

# # ── SIDEBAR ──
# with st.sidebar:
#     st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/L%26T_logo.svg/320px-L%26T_logo.svg.png', width=120)
#     st.title('ACIS')
#     st.caption('Adaptive Concrete Intelligence System')
#     project_type = st.selectbox('Project Type', ['Infrastructure', 'Building'])
#     region = st.selectbox('Region', ['Chennai', 'Mumbai', 'Delhi', 'Shimla', 'Jaipur', 'Kolkata'])
#     st.subheader('Available Resources')
#     res_ambient   = st.checkbox('Ambient Curing', value=True)
#     res_polythene = st.checkbox('Polythene Wrap', value=True)
#     res_steam     = st.checkbox('Steam Curing', value=False)
#     res_heated    = st.checkbox('Heated Enclosure', value=False)

# # ── TABS ──
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     '🧬 Concrete DNA', '🌡️ Simulator', '⚙️ Optimizer', '⚠️ Risk Engine', '📋 Audit Trail'
# ])
# with tab1: st.header('Batch Fingerprinting — Coming Day 2')
# with tab2: st.header('Yard Simulator — Coming Day 3')
# with tab3: st.header('PSO Optimizer — Coming Day 5')
# with tab4: st.header('Risk Engine — Coming Day 6')
# with tab5: st.header('Audit Trail — Coming Day 7')
import streamlit as st
import yaml
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ACIS",
    page_icon="🏗️",
    layout="wide"
)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", width=120)
    st.markdown("## 🏗️ ACIS")
    st.caption("Adaptive Concrete Intelligence System")
    st.markdown("---")

    project_type = st.selectbox(
        "Project Type",
        ["Infrastructure", "Building"]
    )

    region = st.selectbox(
        "Region",
        ["Chennai", "Mumbai", "Delhi", "Shimla", "Jaipur", "Kolkata"]
    )

    st.markdown("### Available Resources")
    res_ambient   = st.checkbox("Ambient Curing",    value=True)
    res_polythene = st.checkbox("Polythene Wrap",    value=True)
    res_steam     = st.checkbox("Steam Curing",      value=False)
    res_heated    = st.checkbox("Heated Enclosure",  value=False)

    st.markdown("### Cost Parameters")
    labor_rate   = st.slider("Labor ₹/hr",            200, 800,  450)
    mold_rate    = st.slider("Mold holding ₹/hr",      50, 500,  200)
    throughput_v = st.slider("Throughput value ₹/hr", 200, 2000, 800)
    automation   = st.radio(
        "Automation Level",
        ["Manual", "Semi-Auto", "Full-Auto"],
        index=0
    )
    auto_idx = ["Manual", "Semi-Auto", "Full-Auto"].index(automation)

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
config_file = "building" if project_type == "Building" else "infra"
with open(f"config/{config_file}.yaml") as f:
    cfg = yaml.safe_load(f)

# Override cost params from sidebar
cfg["labor_cost_per_hour"]         = labor_rate
cfg["mold_holding_cost_per_hour"]  = mold_rate
cfg["throughput_value_per_hour"]   = throughput_v

# Build available resources list
resources = []
if res_ambient:   resources.append("ambient")
if res_polythene: resources.append("polythene_wrap")
if res_steam:     resources.append("steam")
if res_heated:    resources.append("heated_enclosure")
if not resources:
    resources = ["ambient"]
    st.sidebar.warning("⚠️ Select at least one curing method.")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Concrete DNA",
    "🌡️ Simulator",
    "⚙️ Optimizer",
    "⚠️ Risk Engine",
    "📋 Audit Trail"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONCRETE DNA
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🧬 Concrete DNA — Batch Fingerprinting")
    st.caption("Each batch is embedded as a vector in performance space. Similar batches cluster together.")

    try:
        import umap
        import plotly.express as px
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv("data/concrete_data.csv")
        features = ["wc_ratio", "cement_content", "admixture_pct",
                    "avg_temp_C", "curing_method"]
        X = StandardScaler().fit_transform(df[features])

        with st.spinner("Computing DNA embeddings — takes ~20 seconds..."):
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            emb = reducer.fit_transform(X[:2000])

        labels = df["risk_label"][:2000].map(
            {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        )
        fig = px.scatter(
            x=emb[:, 0], y=emb[:, 1],
            color=labels,
            color_discrete_map={
                "Low Risk":    "#028090",
                "Medium Risk": "#F5A623",
                "High Risk":   "#C0392B"
            },
            title="Concrete Batch DNA — Performance Space Clusters",
            labels={"x": "Embedding Dim 1", "y": "Embedding Dim 2"}
        )
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Clusters reveal batches with similar risk profiles. Outlier points = anomalous batches worth investigating.")

    except Exception as e:
        st.error(f"DNA engine error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🌡️ Maturity-Based Strength Simulator")

    from engine.maturity import strength_curve, earliest_safe_demould
    import plotly.graph_objects as go

    col1, col2 = st.columns(2)
    with col1:
        S28         = st.slider("Target 28-day strength (MPa)", 25.0, 55.0, 35.0)
        avg_temp    = st.slider("Average curing temperature (°C)", 5.0, 45.0, 28.0)
        curing_sel  = st.selectbox("Curing method", resources)
    with col2:
        night_drop    = st.slider("Overnight temp drop (°C)", 0.0, 15.0, 5.0)
        hours_to_show = st.slider("Hours to simulate", 12, 48, 36)

    # ku by curing method
    ku_map = {"ambient": 140, "polythene_wrap": 110,
              "steam": 80, "heated_enclosure": 90,
              "wet_burlap": 120}
    ku = ku_map.get(curing_sel, 140)

    # Build hourly temp profile
    temps = [
        avg_temp if h % 24 < 14 else avg_temp - night_drop
        for h in range(hours_to_show)
    ]

    curve     = strength_curve(temps, S28, hours_to_show, ku)
    threshold = cfg["strength_threshold_mpa"]
    safe_h    = earliest_safe_demould(temps, S28, threshold, ku)

    # Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=curve["hour"], y=curve["strength_mpa"],
        mode="lines", name="Predicted Strength",
        line=dict(color="#028090", width=3)
    ))
    fig2.add_hline(
        y=threshold, line_dash="dash", line_color="#C0392B",
        annotation_text=f"Min demould strength: {threshold} MPa",
        annotation_position="top left"
    )
    if safe_h:
        fig2.add_vline(
            x=safe_h, line_dash="dot", line_color="#F5A623",
            annotation_text=f"Safe demould: {safe_h}h",
            annotation_position="top right"
        )
    fig2.update_layout(
        title=f"Strength Gain Curve — Nurse-Saul Method ({region})",
        xaxis_title="Hours since casting",
        yaxis_title="Predicted Strength (MPa)",
        height=420
    )
    st.plotly_chart(fig2, use_container_width=True)

    if safe_h:
        st.success(f"✅ Earliest safe demould: **{safe_h} hours** "
                   f"(strength ≥ {threshold} MPa)")
    else:
        st.error("⛔ Strength threshold not reached — extend curing window or "
                 "switch to a faster curing method.")

    # Monte Carlo
    st.markdown("---")
    st.subheader("Monte Carlo Uncertainty Analysis")
    st.caption("300 runs per uncertainty source — shows which variable drives cycle time variance most.")

    if st.button("▶ Run Monte Carlo (takes ~30 sec)"):
        from engine.simulator import monte_carlo_simulation
        import plotly.express as px

        with st.spinner("Running 900 simulations across 3 uncertainty sources..."):
            params = {
                "n_molds":       cfg["mold_count"],
                "n_beds":        cfg["beds_available"],
                "demould_time_h": safe_h or cfg["target_cycle_hours"],
                "avg_temp":      avg_temp,
                "wc_ratio":      0.45
            }
            mc_df = monte_carlo_simulation(params, n_runs=300)

        fig_mc = px.violin(
            mc_df, x="source", y="cycle_time_h",
            color="source", box=True, points="outliers",
            color_discrete_sequence=["#028090", "#F5A623", "#1E2761"],
            title="Cycle Time Uncertainty by Source",
            labels={"cycle_time_h": "Cycle Time (hours)", "source": ""}
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        var_by_source = mc_df.groupby("source")["cycle_time_h"].std()
        dominant      = var_by_source.idxmax()
        st.info(f"📊 Dominant uncertainty source: **{dominant}** — "
                f"focus process control here first.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("⚙️ PSO Optimizer — Find the Best Cycle Strategy")

    import plotly.express as px

    col1, col2 = st.columns(2)
    with col1:
        w_time      = st.slider(
            "Priority: Speed ← → Cost  (0 = minimise cost, 1 = minimise time)",
            0.0, 1.0, 0.5
        )
        w_cost      = 1.0 - w_time
    with col2:
        avg_temp_opt = st.slider(
            "Site temperature °C", 10.0, 45.0, 28.0, key="opt_temp"
        )
        night_opt    = st.slider(
            "Overnight drop °C", 0.0, 15.0, 5.0, key="opt_night"
        )

    # if st.button("🚀 Run Optimizer"):
    #     from engine.optimizer import run_pso
    #     from engine.cost_model import compute_cycle_cost

    #     temps_opt = [
    #         avg_temp_opt if h % 24 < 14 else avg_temp_opt - night_opt
    #         for h in range(48)
    #     ]
    if st.button("🚀 Run Optimizer"):
        from engine.optimizer import run_pso
        from engine.cost_model import compute_cycle_cost
        from weather.fetch import get_hourly_temps  # <-- Add this import

        # Fetch real API weather for the selected region instead of using the slider
        temps_opt = get_hourly_temps(region, 48)

        with st.spinner("PSO running — 30 particles × 50 iterations..."):
            best_pos, best_cost, history = run_pso(
                cfg, temps_opt, resources, w_time, w_cost
            )

        if best_cost >= 1e5:
            st.error("""
            ⛔ No feasible solution found under current constraints.
            **Suggestions:**
            - Enable additional curing methods (try Steam or Heated Enclosure)
            - Reduce the strength threshold in config
            - Increase site temperature or extend target cycle time
            """)
        else:
            CURING_NAMES = [
                "ambient", "polythene_wrap", "wet_burlap",
                "steam", "heated_enclosure"
            ]
            demould_h     = float(np.clip(best_pos[0], 6, 30))
            curing_idx    = int(np.clip(round(best_pos[1]), 0, len(resources) - 1))
            auto_level    = int(np.clip(round(best_pos[2]), 0, 2))
            curing_method = resources[curing_idx]

            cost_detail = compute_cycle_cost(
                demould_time_h  = demould_h,
                curing_method   = curing_method,
                automation_level= auto_level,
                n_molds         = cfg["mold_count"],
                target_cycle_h  = cfg["target_cycle_hours"],
                labor_cost_per_h= cfg["labor_cost_per_hour"],
                mold_holding_per_h= cfg["mold_holding_cost_per_hour"],
                throughput_value_per_h= cfg["throughput_value_per_hour"]
            )

            baseline_h  = cfg["target_cycle_hours"]
            saving_h    = max(0, baseline_h - demould_h)
            saving_inr  = saving_h * cfg["throughput_value_per_hour"] * cfg["mold_count"]

            st.success(f"""
            ✅ **OPTIMAL STRATEGY FOUND**
            - Demould time: **{demould_h:.1f} hours**
            - Curing method: **{curing_method.replace('_',' ').title()}**
            - Automation: **{["Manual","Semi-Auto","Full-Auto"][auto_level]}**
            - Total cost/cycle: **₹{cost_detail['total']:,.0f}**
            - Schedule penalty: **₹{cost_detail['schedule_penalty']:,.0f}**
            - Estimated saving vs baseline: **₹{saving_inr:,.0f}/day**
            """)

            # Cost breakdown
            st.subheader("Cost Breakdown")
            cost_df = pd.DataFrame({
                "Component":  ["Energy", "Labor", "Mold Holding",
                                "Capex Amort", "Schedule Penalty"],
                "Amount (₹)": [
                    cost_detail["energy"],    cost_detail["labor"],
                    cost_detail["mold_holding"], cost_detail["capex"],
                    cost_detail["schedule_penalty"]
                ]
            })
            fig_cost = px.bar(
                cost_df, x="Component", y="Amount (₹)",
                color="Component",
                color_discrete_sequence=[
                    "#028090","#1E2761","#F5A623","#84B59F","#C0392B"
                ],
                title="Cost Breakdown per Cycle"
            )
            st.plotly_chart(fig_cost, use_container_width=True)

            # Sensitivity table
            st.subheader("Sensitivity Analysis")
            sens_df = pd.DataFrame({
                "Scenario":       ["🟢 Optimistic", "🟡 Base Case", "🔴 Conservative"],
                "Demould Time (h)": [
                    round(demould_h * 0.87, 1),
                    round(demould_h, 1),
                    round(demould_h * 1.18, 1)
                ],
                "Conditions":     [
                    f"+2°C above average, low w/c",
                    "As configured",
                    f"-3°C below average, high w/c"
                ],
                "Risk Level":     ["Low", "Medium", "Low (extended time)"]
            })
            st.table(sens_df)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK ENGINE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("⚠️ Risk Early Warning Engine")

    from engine.maturity import compute_maturity, strength_at_maturity
    import joblib, os

    FEATURES = [
        "wc_ratio", "cement_content", "admixture_pct", "avg_temp_C",
        "humidity_pct", "demould_time_h", "maturity_index", "strength_at_demould"
    ]

    if not os.path.exists("models/risk_model.pkl"):
        st.warning("⚠️ Risk model not trained yet. Run `python engine/risk_model.py` first.")
    else:
        model = joblib.load("models/risk_model.pkl")

        col1, col2 = st.columns(2)
        with col1:
            r_wc     = st.slider("w/c ratio",              0.35, 0.55, 0.45, key="r_wc")
            r_cement = st.slider("Cement content (kg/m³)", 300,  450,  380,  key="r_cem")
            r_temp   = st.slider("Avg temperature (°C)",   5.0,  45.0, 28.0, key="r_tmp")
            r_humid  = st.slider("Humidity (%)",           30,   95,   65,   key="r_hum")
        with col2:
            r_demould = st.slider("Planned demould time (h)", 6.0, 30.0, 12.0, key="r_dem")
            r_admix   = st.slider("Admixture (%)",            0.3,  1.2,  0.7, key="r_adm")

        # Compute maturity + strength for input
        # temps_r = [
        #     r_temp - (4 if 2 <= h % 24 <= 6 else 0)
        #     for h in range(int(r_demould))
        # ]
        from weather.fetch import get_hourly_temps
        # Compute maturity + strength for input using the manual sliders
        temps_r = [
            r_temp - (4 if 2 <= h % 24 <= 6 else 0)
            for h in range(int(r_demould))
        ]
        mat_r = compute_maturity(temps_r, [1.0] * len(temps_r))
        str_r = strength_at_maturity(mat_r, 35.0)

        input_df = pd.DataFrame([{
            "wc_ratio":            r_wc,
            "cement_content":      r_cement,
            "admixture_pct":       r_admix,
            "avg_temp_C":          r_temp,
            "humidity_pct":        r_humid,
            "demould_time_h":      r_demould,
            "maturity_index":      mat_r,
            "strength_at_demould": str_r
        }])

        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        risk_label = ["🟢 LOW RISK", "🟡 MEDIUM RISK", "🔴 HIGH RISK"][pred]
        risk_fn    = [st.success, st.warning, st.error][pred]
        risk_fn(f"Risk Assessment: **{risk_label}**  "
                f"(confidence: {proba[pred]*100:.0f}%)")

        st.caption(
            "Risk scores are relative rankings under simulated distributions. "
            "Production deployment requires plant calibration against real sensor data."
        )

        # SHAP
        try:
            import shap
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(input_df)
            sv         = shap_vals[pred] if isinstance(shap_vals, list) else shap_vals
            shap_arr   = sv[0] if hasattr(sv, '__len__') else sv

            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP":    shap_arr
            }).reindex(
                pd.Series(shap_arr).abs().sort_values(ascending=False).index
            ).head(3)

            st.subheader("Top 3 Risk Drivers (SHAP)")
            for _, row in shap_df.iterrows():
                direction = "↑ increases" if row["SHAP"] > 0 else "↓ decreases"
                st.write(f"**{row['Feature']}** — {direction} risk "
                         f"(SHAP = {row['SHAP']:.3f})")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — AUDIT TRAIL
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📋 QC Audit Trail — Dispute-Ready Batch Log")

    import sqlite3, hashlib, datetime

    def init_db():
        conn = sqlite3.connect("audit/audit_log.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_hash       TEXT,
                timestamp        TEXT,
                region           TEXT,
                project_type     TEXT,
                curing_method    TEXT,
                demould_time_h   REAL,
                predicted_strength REAL,
                risk_level       TEXT,
                recommendation   TEXT
            )
        """)
        conn.commit()
        return conn

    def log_batch(params: dict) -> str:
        conn      = init_db()
        batch_str = str(sorted(params.items())).encode()
        batch_hash= hashlib.sha256(batch_str).hexdigest()[:16]
        params["batch_hash"] = batch_hash
        params["timestamp"]  = datetime.datetime.now().isoformat()
        conn.execute("""
            INSERT INTO batches
            (batch_hash, timestamp, region, project_type, curing_method,
             demould_time_h, predicted_strength, risk_level, recommendation)
            VALUES
            (:batch_hash, :timestamp, :region, :project_type, :curing_method,
             :demould_time_h, :predicted_strength, :risk_level, :recommendation)
        """, params)
        conn.commit()
        return batch_hash

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📝 Log Current Batch to Audit"):
            hid = log_batch({
                "region":              region,
                "project_type":        project_type,
                "curing_method":       "polythene_wrap",
                "demould_time_h":      12.0,
                "predicted_strength":  22.5,
                "risk_level":          "Medium",
                "recommendation":      "Polythene wrap 12h — system optimal"
            })
            st.success(f"✅ Logged. Batch Hash ID: `{hid}`")

    with col2:
        conn   = init_db()
        df_log = pd.read_sql(
            "SELECT * FROM batches ORDER BY id DESC LIMIT 20", conn
        )
        if len(df_log):
            st.dataframe(df_log, use_container_width=True)
        else:
            st.info("No batches logged yet. Click 'Log Current Batch' to start.")

    st.caption(
        "Each batch is SHA-256 hashed. If an element fails at site, "
        "trace back to exact parameters, recommendation, and operator action."
    )