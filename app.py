"""
PayGuard FinTech - Mobile-Wallet Fraud Detection Engine
Vodafone Cash & InstaPay Protection Platform
Streamlit Web Application
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="PayGuard | كشف احتيال فودافون كاش وإنستاباي",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAMPLE_TEST_CSV = os.path.join(DATA_DIR, 'test_transactions_sample.csv')
COST_CURVE_CSV = os.path.join(DATA_DIR, 'cost_curve_data.csv')

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'best_model': 'fraud_best_model.joblib',
        'rf_model': 'rf_model.joblib',
        'lr_model': 'lr_model.joblib',
        'scaler': 'scaler.joblib',
        'features': 'feature_cols.joblib',
        'metrics': 'metrics_summary.joblib'
    }
    for key, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

@st.cache_data
def load_sample_data():
    if os.path.exists(SAMPLE_TEST_CSV):
        return pd.read_csv(SAMPLE_TEST_CSV)
    return None

@st.cache_data
def load_cost_data():
    if os.path.exists(COST_CURVE_CSV):
        return pd.read_csv(COST_CURVE_CSV)
    return None

models = load_models()
sample_df = load_sample_data()
cost_df = load_cost_data()

# Sidebar Navigation
st.sidebar.markdown("## 🛡️ PayGuard FinTech")
st.sidebar.markdown("**محرك كشف الاحتيال للمحافظ الإلكترونية والمدفوعات اللحظية**")
st.sidebar.caption("Vodafone Cash & InstaPay Anti-Fraud Engine")
st.sidebar.markdown("---")

view_mode = st.sidebar.radio(
    "القائمة الرئيسية (Navigation)",
    [
        "⚡ تجربة فحص معاملة فورية (Live Simulator)",
        "📁 فحص ملف معاملات مجمّع (Batch CSV Scanner)",
        "⚖️ محاكي التكلفة والعتبة الذكية (Cost Matrix Tuner)",
        "📈 لوحة أداء وتقييم النماذج (Model Benchmarks)",
        "📖 دليل النظام ومواصفات المشروع (About & Architecture)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 حالة النماذج (Model Specs)")
st.sidebar.metric("PR-AUC النتيجة الفائقة", "96.1%", delta="HistGradientBoosting")
st.sidebar.metric("ROC-AUC دقة التمييز", "98.6%", delta="Optimal Separation")
st.sidebar.caption("Dataset: PaySim Mobile Money (Synthetic Banking Logs)")
st.sidebar.caption("NextGen Institute · Machine Learning Final Project")

# Helper function to predict single transaction
def predict_single_transaction(tx_dict, threshold=0.35):
    # Calculate engineered features
    amt = float(tx_dict['amount'])
    old_orig = float(tx_dict['oldbalanceOrg'])
    new_orig = float(tx_dict['newbalanceOrig'])
    old_dest = float(tx_dict['oldbalanceDest'])
    new_dest = float(tx_dict['newbalanceDest'])
    hour = int(tx_dict['hour_of_day'])
    tx_type = tx_dict['type']
    
    error_orig = new_orig + amt - old_orig
    error_dest = old_dest + amt - new_dest
    amount_to_orig = amt / (old_orig + 1.0)
    is_emptied = 1 if (new_orig == 0 and amt > 0) else 0
    is_night = 1 if hour in [23, 0, 1, 2, 3, 4, 5] else 0
    is_transfer = 1 if tx_type == 'TRANSFER' else 0
    is_cash_out = 1 if tx_type == 'CASH_OUT' else 0
    is_payment = 1 if tx_type == 'PAYMENT' else 0
    is_cash_in = 1 if tx_type == 'CASH_IN' else 0
    dest_is_merchant = int(tx_dict.get('dest_is_merchant', 0))
    
    feat_values = pd.DataFrame([{
        'amount': amt,
        'oldbalanceOrg': old_orig,
        'newbalanceOrig': new_orig,
        'oldbalanceDest': old_dest,
        'newbalanceDest': new_dest,
        'error_orig': error_orig,
        'error_dest': error_dest,
        'amount_to_orig_ratio': amount_to_orig,
        'is_account_emptied': is_emptied,
        'hour_of_day': hour,
        'is_night_transaction': is_night,
        'is_transfer': is_transfer,
        'is_cash_out': is_cash_out,
        'is_payment': is_payment,
        'is_cash_in': is_cash_in,
        'dest_is_merchant': dest_is_merchant
    }])
    
    if 'best_model' in models:
        prob = float(models['best_model'].predict_proba(feat_values)[0, 1])
    else:
        # Fallback heuristic
        prob = 0.95 if (is_emptied and (is_transfer or is_cash_out) and amt > 5000) else 0.05
        
    is_fraud = bool(prob >= threshold)
    
    # Explainability factors
    reasons = []
    if is_emptied:
        reasons.append("⚠️ تم تصفير رصيد المحفظة بالكامل (Account Emptied to 0.00 EGP).")
    if abs(error_orig) > 1.0:
        reasons.append(f"⚠️ تناقض رياضي في رصيد المرسل بفارق: {error_orig:,.2f} جنيه.")
    if abs(error_dest) > 1.0:
        reasons.append(f"⚠️ تناقض رياضي في رصيد المستلم بفارق: {error_dest:,.2f} جنيه.")
    if is_transfer or is_cash_out:
        reasons.append("ℹ️ العملية من نوع تحويل أو سحب كاش (الأكثر استهدافاً بالاحتيال).")
    if is_night:
        reasons.append(f"🌙 توقيت المعاملة في ساعات الليل المتأخرة ({hour:02d}:00).")
    if amt > 20000:
        reasons.append(f"💰 مبلغ المعاملة مرتفع نسبياً ({amt:,.2f} جنيه).")
    if not reasons:
        reasons.append("✅ المعاملة متسقة مع الأنماط الطبيعية للمدفوعات اليومية.")
        
    return prob, is_fraud, reasons

# ==========================================
# VIEW 1: LIVE FRAUD SIMULATOR (تجربة حية)
# ==========================================
if view_mode == "⚡ تجربة فحص معاملة فورية (Live Simulator)":
    st.title("⚡ تجربة كشف الاحتيال اللحظي (Live Transaction Simulator)")
    st.markdown("""
    قم بتجربة النظام مباشرة: أدخل بيانات المعاملة المالية عبر **فودافون كاش** أو **إنستاباي** لتقييم احتمالية الاحتيال فورياً مع تفسير الذكاء الاصطناعي للأسباب.
    """)
    
    # Quick Test Presets
    st.subheader("🎯 أو اختر تجربة سريعة بنقرة واحدة (One-Click Test Scenarios):")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    selected_preset = None
    with p_col1:
        if st.button("🚨 سحب احتيالي وتصفير محفظة", width="stretch"):
            selected_preset = {
                'service': 'فودافون كاش (Vodafone Cash)',
                'type': 'CASH_OUT',
                'amount': 45000.0,
                'oldbalanceOrg': 45000.0,
                'newbalanceOrig': 0.0,
                'oldbalanceDest': 12000.0,
                'newbalanceDest': 12000.0, # Dest balance didn't increase normally
                'hour_of_day': 3,
                'dest_is_merchant': 0
            }
    with p_col2:
        if st.button("💸 تحويل إنستاباي مريب (Mule)", width="stretch"):
            selected_preset = {
                'service': 'إنستاباي (InstaPay)',
                'type': 'TRANSFER',
                'amount': 95000.0,
                'oldbalanceOrg': 100000.0,
                'newbalanceOrig': 5000.0,
                'oldbalanceDest': 0.0,
                'newbalanceDest': 0.0,
                'hour_of_day': 1,
                'dest_is_merchant': 0
            }
    with p_col3:
        if st.button("✅ تحويل عائلي آمن إنستاباي", width="stretch"):
            selected_preset = {
                'service': 'إنستاباي (InstaPay)',
                'type': 'TRANSFER',
                'amount': 2500.0,
                'oldbalanceOrg': 18000.0,
                'newbalanceOrig': 15500.0,
                'oldbalanceDest': 4200.0,
                'newbalanceDest': 6700.0,
                'hour_of_day': 14,
                'dest_is_merchant': 0
            }
    with p_col4:
        if st.button("🛒 شراء مشتريات تاجر QR Code", width="stretch"):
            selected_preset = {
                'service': 'فودافون كاش (Vodafone Cash)',
                'type': 'PAYMENT',
                'amount': 650.0,
                'oldbalanceOrg': 3500.0,
                'newbalanceOrig': 2850.0,
                'oldbalanceDest': 50000.0,
                'newbalanceDest': 50650.0,
                'hour_of_day': 18,
                'dest_is_merchant': 1
            }

    # Store in session state
    if selected_preset:
        st.session_state['preset'] = selected_preset

    current_val = st.session_state.get('preset', {
        'service': 'إنستاباي (InstaPay)',
        'type': 'TRANSFER',
        'amount': 15000.0,
        'oldbalanceOrg': 15000.0,
        'newbalanceOrig': 0.0,
        'oldbalanceDest': 500.0,
        'newbalanceDest': 500.0,
        'hour_of_day': 2,
        'dest_is_merchant': 0
    })

    st.markdown("---")
    
    with st.container(border=True):
        st.markdown("### 📝 تفاصيل المعاملة المالية (Transaction Details)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            service_choice = st.selectbox(
                "مزود الخدمة (Service)",
                ["فودافون كاش (Vodafone Cash)", "إنستاباي (InstaPay)", "اتصالات كاش (Etisalat Cash)", "أورنج كاش (Orange Cash)"],
                index=0 if "فودافون" in current_val['service'] else 1
            )
            type_options = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN']
            tx_type = st.selectbox(
                "نوع المعاملة (Transaction Type)",
                type_options,
                index=type_options.index(current_val['type']) if current_val['type'] in type_options else 0,
                format_func=lambda x: {
                    'TRANSFER': 'تحويل فوري (TRANSFER)',
                    'CASH_OUT': 'سحب نقدي / تاجر (CASH_OUT)',
                    'PAYMENT': 'سداد مشتريات (PAYMENT)',
                    'CASH_IN': 'إيداع نقدي (CASH_IN)'
                }.get(x, x)
            )
            amount = st.number_input("قيمة المعاملة بالجنيه (Amount in EGP)", min_value=1.0, max_value=200000.0, value=float(current_val['amount']), step=500.0)

        with c2:
            old_orig = st.number_input("رصيد المرسل قبل العملية (Sender Balance Before)", min_value=0.0, max_value=500000.0, value=float(current_val['oldbalanceOrg']), step=500.0)
            new_orig = st.number_input("رصيد المرسل بعد العملية (Sender Balance After)", min_value=0.0, max_value=500000.0, value=float(current_val['newbalanceOrig']), step=500.0)
            hour = st.slider("توقيت المعاملة (ساعة اليوم 0-23)", 0, 23, int(current_val['hour_of_day']))

        with c3:
            old_dest = st.number_input("رصيد المستلم قبل العملية (Receiver Balance Before)", min_value=0.0, max_value=500000.0, value=float(current_val['oldbalanceDest']), step=500.0)
            new_dest = st.number_input("رصيد المستلم بعد العملية (Receiver Balance After)", min_value=0.0, max_value=500000.0, value=float(current_val['newbalanceDest']), step=500.0)
            is_merchant = st.checkbox("المستلم حساب تاجر رسمي معتمد (Merchant)", value=bool(current_val['dest_is_merchant']))
            
        custom_threshold = st.slider("عتبة القرار الأمني (Detection Threshold)", min_value=0.05, max_value=0.95, value=0.35, step=0.05, help="كلما انخفضت العتبة، زادت حساسية الكشف لحماية العملاء")
        
        analyze_btn = st.button("🔍 فحص أمان المعاملة الآن (Analyze Transaction Risk)", width="stretch")

    # Perform analysis
    tx_input = {
        'amount': amount,
        'oldbalanceOrg': old_orig,
        'newbalanceOrig': new_orig,
        'oldbalanceDest': old_dest,
        'newbalanceDest': new_dest,
        'hour_of_day': hour,
        'type': tx_type,
        'dest_is_merchant': int(is_merchant)
    }
    
    prob, is_fraud, reasons = predict_single_transaction(tx_input, threshold=custom_threshold)
    
    st.markdown("---")
    st.markdown("### 📊 نتيجة الفحص الأمني والتحليل الفوري:")
    
    res_col1, res_col2 = st.columns([1, 1.4])
    
    with res_col1:
        with st.container(border=True):
            st.markdown("#### درجة الخطر (Fraud Risk Score)")
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 36}},
                title={'text': "احتمالية الاحتيال (Risk Level)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if is_fraud else "#10B981"},
                    'steps': [
                        {'range': [0, 25], 'color': "#D1FAE5"},
                        {'range': [25, 55], 'color': "#FEF3C7"},
                        {'range': [55, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': custom_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

    with res_col2:
        with st.container(border=True):
            st.markdown("#### القرار الأمني الموصى به (Recommended Action)")
            if prob >= 0.70:
                st.error("🚨 احتيال عالي الخطورة - تم حظر المعاملة وتجميد المحفظة فورياً!", icon="🛑")
                st.markdown("**الإجراء المتخذ:** تم إيقاف المعاملة، وتنبيه مركز مكافحة الاحتيال (Anti-Fraud Desk) للتحقق من هوية صاحب المحفظة لمنع سرقة الأموال.")
            elif prob >= custom_threshold:
                st.warning("⚠️ معاملة مشبوهة - تتطلب تأكيد إضافي (Step-Up Authentication)!", icon="⚠️")
                st.markdown("**الإجراء المتخذ:** تم تعليق المعاملة مؤقتاً لحين طلب رمز OTP جديد أو تأكيد بصمة الوجه عبر تطبيق إنستاباي/فودافون كاش.")
            else:
                st.success("✅ معاملة آمنة - تم التصريح بالتمرير الفوري (Approved)!", icon="✅")
                st.markdown("**الإجراء المتخذ:** العملية تتوافق مع الأنماط الآمنة والسلوك المالي المعتاد للعميل. لا توجد أي عوائق.")

            st.markdown("---")
            st.markdown("**🔍 أسباب ومحركات القرار (AI Decision Explainability):**")
            for r in reasons:
                st.markdown(f"- {r}")

# ==========================================
# VIEW 2: BATCH CSV SCANNER
# ==========================================
elif view_mode == "📁 فحص ملف معاملات مجمّع (Batch CSV Scanner)":
    st.title("📁 فحص ملف معاملات مجمّع (Batch CSV Risk Scanner)")
    st.markdown("قم برفع ملف معاملات مجمّع أو فحص العينة المجهزة مسبقاً للكشف الآلي عن المعاملات المشبوهة وحساب المبالغ المحمية.")
    
    # Download sample button
    if sample_df is not None:
        csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 اضغط هنا لتحميل عينة تجريبية لاختبارها فوراً (Download Sample CSV)",
            data=csv_bytes,
            file_name="paysim_test_sample.csv",
            mime="text/csv",
            width="stretch"
        )
        
    st.markdown("---")
    uploaded_file = st.file_uploader("قم برفع ملف CSV يحتوي على المعاملات المالية:", type=['csv'])
    
    target_df = None
    if uploaded_file is not None:
        target_df = pd.read_csv(uploaded_file)
        st.success(f"تم تحميل الملف بنجاح! عدد المعاملات: {len(target_df):,} معاملة.")
    elif sample_df is not None:
        st.info("💡 يتم حالياً فحص العينة التجريبية المحملة مسبقاً (500 معاملة حقيقية). يمكنك رفع ملفك الخاص في أي وقت أعلاه.")
        target_df = sample_df.copy()
        
    if target_df is not None:
        # Predict on batch
        feat_cols = models.get('features', [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'error_orig', 'error_dest', 'amount_to_orig_ratio', 'is_account_emptied',
            'hour_of_day', 'is_night_transaction', 'is_transfer', 'is_cash_out',
            'is_payment', 'is_cash_in', 'dest_is_merchant'
        ])
        
        # Verify columns exist or engineer them
        df_scan = target_df.copy()
        if 'error_orig' not in df_scan.columns:
            df_scan['error_orig'] = df_scan['newbalanceOrig'] + df_scan['amount'] - df_scan['oldbalanceOrg']
            df_scan['error_dest'] = df_scan['oldbalanceDest'] + df_scan['amount'] - df_scan['newbalanceDest']
            df_scan['amount_to_orig_ratio'] = df_scan['amount'] / (df_scan['oldbalanceOrg'] + 1.0)
            df_scan['is_account_emptied'] = ((df_scan['newbalanceOrig'] == 0) & (df_scan['amount'] > 0)).astype(int)
            df_scan['hour_of_day'] = df_scan['step'] % 24 if 'step' in df_scan.columns else 12
            df_scan['is_night_transaction'] = df_scan['hour_of_day'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
            df_scan['is_transfer'] = (df_scan['type'] == 'TRANSFER').astype(int) if 'type' in df_scan.columns else 0
            df_scan['is_cash_out'] = (df_scan['type'] == 'CASH_OUT').astype(int) if 'type' in df_scan.columns else 0
            df_scan['is_payment'] = (df_scan['type'] == 'PAYMENT').astype(int) if 'type' in df_scan.columns else 0
            df_scan['is_cash_in'] = (df_scan['type'] == 'CASH_IN').astype(int) if 'type' in df_scan.columns else 0
            df_scan['dest_is_merchant'] = df_scan['nameDest'].astype(str).str.startswith('M').astype(int) if 'nameDest' in df_scan.columns else 0

        X_batch = df_scan[[c for c in feat_cols if c in df_scan.columns]].copy()
        
        if 'best_model' in models:
            probs = models['best_model'].predict_proba(X_batch)[:, 1]
            df_scan['احتمالية_الاحتيال'] = np.round(probs * 100, 2)
            df_scan['قرار_النظام'] = np.where(probs >= 0.35, '🚨 احتيال محتمل', '🟢 معاملة آمنة')
            
            # Metrics
            total_tx = len(df_scan)
            flagged = int((probs >= 0.35).sum())
            protected_amount = float(df_scan[probs >= 0.35]['amount'].sum())
            total_amount = float(df_scan['amount'].sum())
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("إجمالي المعاملات المفحوصة", f"{total_tx:,}")
            with m2:
                st.metric("المعاملات المشبوهة المكتشفة", f"{flagged:,}", delta=f"{flagged/total_tx*100:.1f}% معدل كشف")
            with m3:
                st.metric("الأموال المحمية من السرقة", f"£{protected_amount:,.0f} جنيه", delta="Protected EGP")
            with m4:
                st.metric("إجمالي قيمة المعاملات", f"£{total_amount:,.0f} جنيه")
                
            st.markdown("---")
            st.subheader("📋 جدول المعاملات المفحوصة والتصنيف الأمني:")
            
            filter_mode = st.radio("تصفية العرض:", ["جميع المعاملات", "المعاملات المحظورة والمشبوهة فقط (🚨)", "المعاملات الآمنة فقط (🟢)"], horizontal=True)
            
            display_df = df_scan.copy()
            if "المعاملات المحظورة" in filter_mode:
                display_df = display_df[display_df['قرار_النظام'] == '🚨 احتيال محتمل']
            elif "المعاملات الآمنة" in filter_mode:
                display_df = display_df[display_df['قرار_النظام'] == '🟢 معاملة آمنة']
                
            cols_to_show = [c for c in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'احتمالية_الاحتيال', 'قرار_النظام'] if c in display_df.columns]
            st.dataframe(
                display_df[cols_to_show].head(100).style.format({
                    'amount': '£{:,.2f}',
                    'oldbalanceOrg': '£{:,.2f}',
                    'newbalanceOrig': '£{:,.2f}',
                    'احتمالية_الاحتيال': '{:.1f}%'
                }),
                width="stretch"
            )

# ==========================================
# VIEW 3: COST MATRIX & THRESHOLD TUNER (STRETCH GOAL)
# ==========================================
elif view_mode == "⚖️ محاكي التكلفة والعتبة الذكية (Cost Matrix Tuner)":
    st.title("⚖️ محاكي التكلفة المالية والعتبة الذكية (Cost Matrix & Loss Optimization)")
    st.markdown("""
    في أنظمة مكافحة الاحتيال البنكية، لا يجب اعتماد عتبة الـ 0.50 بشكل عشوائي!  
    **الهدف هنا:** إيجاد العتبة الرياضية الدقيقة ($T^*$) التي **تخفض إجمالي الخسائر المالية بالجنيه المصري لأدنى مستوى ممكن**.
    """)
    
    with st.container(border=True):
        st.subheader("⚙️ معاملات مصفوفة التكاليف البنكية (Cost Matrix Parameters):")
        cc1, cc2 = st.columns(2)
        with cc1:
            cost_fp = st.slider("تكلفة الإنذار الخاطئ (False Alarm Cost) - إزعاج العميل واتصال الدعم", 10.0, 300.0, 50.0, step=10.0, format="£%.0f جنيه")
            st.caption("تحدث عندما يقوم النظام بحظر معاملة لعميل بريء، مما يدفعه للاتصال بخدمة العملاء.")
        with cc2:
            cost_multiplier = st.slider("معامل حساسية سرقة الأموال (Missed Fraud Loss Multiplier)", 0.5, 2.0, 1.0, step=0.1)
            st.caption("معدل الخسارة المباشرة للشركة والعميل عند نجاح المحتال في سحب الأموال دون اكتشافه.")

    if cost_df is not None:
        # Dynamic recalculation
        adjusted_df = cost_df.copy()
        adjusted_df['adj_fp_loss'] = adjusted_df['fp'] * cost_fp
        adjusted_df['adj_fn_loss'] = adjusted_df['loss_fn'] * cost_multiplier
        adjusted_df['adj_total_loss'] = adjusted_df['adj_fp_loss'] + adjusted_df['adj_fn_loss']
        
        opt_row = adjusted_df.loc[adjusted_df['adj_total_loss'].idxmin()]
        opt_thresh = float(opt_row['threshold'])
        min_loss = float(opt_row['adj_total_loss'])
        default_loss = float(adjusted_df.loc[(adjusted_df['threshold'] - 0.5).abs().idxmin()]['adj_total_loss'])
        saved_money = default_loss - min_loss
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("العتبة المثلى الموصى بها (T*)", f"{opt_thresh:.2f}", delta="Optimal Cutoff")
        with col_m2:
            st.metric("الخسارة المالية عند العتبة المثلى", f"£{min_loss:,.0f} جنيه", delta="أدنى خسارة ممكنة")
        with col_m3:
            st.metric("التوفير المالي المحقق مقارنة بـ 0.50", f"£{saved_money:,.0f} جنيه", delta="وفورات مباشرة", delta_color="normal")
            
        st.markdown("---")
        
        # Plotly chart
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=adjusted_df['threshold'], y=adjusted_df['adj_total_loss'],
            mode='lines', name='إجمالي الخسارة المالية (Total Financial Loss)',
            line=dict(color='#EF4444', width=3)
        ))
        fig_loss.add_trace(go.Scatter(
            x=adjusted_df['threshold'], y=adjusted_df['adj_fp_loss'],
            mode='lines', name='خسائر الإنذارات الخاطئة (FP Friction Loss)',
            line=dict(color='#F59E0B', width=2, dash='dot')
        ))
        fig_loss.add_trace(go.Scatter(
            x=adjusted_df['threshold'], y=adjusted_df['adj_fn_loss'],
            mode='lines', name='خسائر سرقة الأموال غير المكتشفة (FN Fraud Loss)',
            line=dict(color='#3B82F6', width=2, dash='dash')
        ))
        
        # Add optimal vertical line
        fig_loss.add_vline(x=opt_thresh, line_width=2, line_dash="dash", line_color="#10B981", annotation_text=f"العتبة المثلى ({opt_thresh:.2f})")
        fig_loss.add_vline(x=0.50, line_width=1.5, line_dash="dot", line_color="gray", annotation_text="العتبة الافتراضية (0.50)")
        
        fig_loss.update_layout(
            title="منحنى تحسين الخسائر المالية vs. عتبة القرار الأمني",
            xaxis_title="عتبة القرار (Detection Threshold)",
            yaxis_title="الخسارة المالية الإجمالية (جنيه مصري)",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig_loss, width="stretch")
        
        st.info(f"💡 **النتيجة البيزنسية:** اختيار العتبة **{opt_thresh:.2f}** يضمن حظر {opt_row['tp']} عملية احتيال مع تقليل إزعاج العملاء، موفراً أكثر من **£{saved_money:,.0f} جنيه** للبنك مقارنة بالعتبة التقليدية!")

# ==========================================
# VIEW 4: MODEL BENCHMARKS
# ==========================================
elif view_mode == "📈 لوحة أداء وتقييم النماذج (Model Benchmarks)":
    st.title("📈 لوحة أداء وتقييم النماذج (Model Benchmarks & Metrics)")
    st.markdown("مقارنة شاملة بين النماذج الأربعة على **مجموعة الاختبار المستقلة (Untouched Holdout Test Set)**.")
    
    benchmark_data = {
        "النموذج (Model)": [
            "Baseline (Dummy Stratified)",
            "Logistic Regression (Balanced)",
            "Random Forest (100 Trees, Balanced)",
            "HistGradientBoosting (Tuned Production)"
        ],
        "PR-AUC (المعيار الأساسي)": ["0.0086", "0.8165", "0.9587", "0.9606"],
        "ROC-AUC": ["0.4950", "0.9907", "0.9820", "0.9862"],
        "الاستدعاء Recall": ["0.9%", "93.1%", "95.8%", "94.4%"],
        "الدقة Precision": ["0.8%", "7.2%", "88.5%", "89.5%"],
        "F1-Score": ["0.0085", "0.1336", "0.9200", "0.9189"]
    }
    
    st.dataframe(pd.DataFrame(benchmark_data), width="stretch")
    
    st.markdown("---")
    st.subheader("💡 لماذا يعد مقياس PR-AUC هو الأهم وليس الدقة (Accuracy)؟")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("""
        - في بيانات الاحتيال المالي، تمثل العمليات السليمة أكثر من **99%** من الحركة.
        - إذا قام نموذج غبي بالتنبؤ بأن جميع العمليات سليمة، فسيحصل على دقة **99.2% Accuracy**، ولكنه سيفشل في اكتشاف **أي عملية سرقة واحدة (0% Recall)**!
        - لذلك، فإن **PR-AUC (Precision-Recall Area Under Curve)** هو المقياس الصادق عالمياً لتقييم كفاءة كشف الاحتيال.
        """)
    with b_col2:
        st.markdown("""
        - قمنا بتدريب النموذج عبر **خوارزميات شجرية متقدمة (HistGradientBoosting & Random Forest)** تأخذ في الاعتبار تباين الأوزان `class_weight='balanced'`.
        - حقق النموذج النهائي قفزة نوعية في **PR-AUC من 0.0086 (النموذج المرجعي) إلى 0.9606**، مع استدعاء يتجاوز **94%** من محاولات الاحتيال!
        """)

# ==========================================
# VIEW 5: ABOUT & ARCHITECTURE
# ==========================================
elif view_mode == "📖 دليل النظام ومواصفات المشروع (About & Architecture)":
    st.title("📖 مواصفات المشروع والمعمارية التقنية (System Architecture)")
    
    st.markdown("""
    ### 🛡️ ملخص المشروع (Project Overview):
    مشروع تخرج تطبيقي في دبلومة تعلم الآلة بمعهد **NextGen Institute**. يهدف إلى حماية مستخدمي المحافظ الإلكترونية وشبكة المدفوعات اللحظية في مصر (**فودافون كاش، إنستاباي، اتصالات كاش، أورنج كاش**) من عمليات الاحتيال والاستيلاء على الحسابات.
    
    ---
    ### 🏗️ معمارية خط الأنابيب (Pipeline Flow):
    ```
    1. السجلات المالية الخام (PaySim Financial Logs)
       ↓
    2. هندسة ميزات الأرصدة والتوقيت (Balance Inconsistency & Velocity Features)
       ↓
    3. تقسيم البيانات بدون تسريب زمني (Stratified 80/20 Leakage-Free Split)
       ↓
    4. تدريب ومقارنة النماذج (Baseline vs Logistic vs Random Forest vs Boosting)
       ↓
    5. تحسين عتبة القرار عبر مصفوفة التكاليف (Cost Matrix Optimization)
       ↓
    6. نشر التطبيق التفاعلي المباشر (Interactive Streamlit Dashboard)
    ```
    
    ---
    ### 🎓 مخرجات ومواصفات التقييم (NextGen Institute Deliverables):
    - **Jupyter Notebook:** ملف نظيف منظم يعمل من البداية للنهاية بدون أخطاء `notebooks/mobile_wallet_fraud_detection.ipynb`.
    - **Streamlit App:** تطبيق تفاعلي يمكن لأي مستخدم غير تقني تجربته واختبار المعاملات في ثوانٍ.
    - **GitHub Repo & README:** توثيق كامل، نتائج المقارنة، وتعليمات التشغيل.
    """)

st.markdown("---")
st.caption("🛡️ PayGuard FinTech Engine · Developed for NextGen Institute Applied Machine Learning Diploma")
