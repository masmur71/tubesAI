import streamlit as st
import json
import pickle
import requests
import numpy as np
import scipy.sparse as sp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Apotek Intelligence_Kelompok 2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. FUNGSI CACHING DATA PKA (ASISTEN AI)
# ==========================================
@st.cache_resource
def load_pka_models():
    try:
        with open('pka_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = sp.load_npz('pka_tfidf_matrix.npz')
        return vectorizer, tfidf_matrix
    except Exception as e:
        return None, None

@st.cache_data
def load_pka_data():
    try:
        with open('pka_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        with open('pka_metadata.json', 'r', encoding='utf-8') as f:
            pka_metadata = json.load(f)
        return chunks, pka_metadata
    except Exception as e:
        return None, None

# ==========================================
# 3. FUNGSI CACHING DATA IAS (DASHBOARD)
# ==========================================
@st.cache_data
def load_ias_data():
    try:
        with open('metadata.json', 'r') as f:
            ias_metadata = json.load(f)
        
        df_sales = pd.read_csv('salesdaily_processed.csv')
        df_sales['datum'] = pd.to_datetime(df_sales['datum'])
        
        df_risk = pd.read_csv('risk_classification.csv')
        
        with open('forecast_results.json', 'r') as f:
            forecast_data = json.load(f)
            
        return ias_metadata, df_sales, df_risk, forecast_data
    except Exception as e:
        return None, None, None, None

# ==========================================
# 4. FUNGSI INTI PKA
# ==========================================
def retrieve(query, vectorizer, tfidf_matrix, chunks, top_k, threshold):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': chunks[idx]['text'],
            'source': chunks[idx]['source'],
            'score': round(float(similarities[idx]), 4)
        })
    
    fallback = results[0]['score'] < threshold if results else True
    return results, fallback

def build_prompt(query, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(f"[Konteks {i} — Sumber: {chunk['source']} | Relevansi: {chunk['score']}]\n{chunk['text']}")
    context_block = "\n\n".join(context_parts)
    return f"Berikut adalah konteks dari dokumen knowledge base apotek:\n\n{context_block}\n\n---\nPertanyaan dari tenaga kefarmasian:\n{query}\n\nBerikan jawaban yang akurat berdasarkan konteks di atas."

def call_groq_api(user_prompt, api_key, model="llama3-70b-8192"):
    system_prompt = """Kamu adalah Pharmacy Knowledge Assistant (PKA).
ATURAN UTAMA:
1. Jawab HANYA berdasarkan konteks dokumen yang diberikan.
2. Jika tidak ada dalam konteks, nyatakan secara eksplisit.
3. Sebutkan nama dokumen sumber di akhir jawaban.
4. Jangan memberikan diagnosis di luar regulasi farmasi."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "temperature": 0.1,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error API: {str(e)}"


# ==========================================
# 5. SIDEBAR & NAVIGASI
# ==========================================

st.sidebar.title("Apotek Intelligence")
st.sidebar.caption("Kelompok 2")

pilihan_fitur = st.sidebar.radio(
    "Pilih Modul:",
    ["📊 IAS (Dashboard Analitik)", "💊 PKA (Asisten AI)"],
    index=0
)
st.sidebar.divider()

# ==========================================
# 6. MODUL 1: IAS (INTELLIGENT ANALYTICS SYSTEM)
# ==========================================
if pilihan_fitur == "📊 IAS (Dashboard Analitik)":
    ias_metadata, df_sales, df_risk, forecast_data = load_ias_data()
    
    if ias_metadata is None:
        st.error("Data IAS tidak ditemukan. Pastikan metadata.json, salesdaily_processed.csv, risk_classification.csv, dan forecast_results.json ada.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Filter IAS")
        selected_atc_trend = st.selectbox(
            "Kategori Obat (Historis):", 
            ias_metadata['atc_cols'], 
            format_func=lambda x: ias_metadata['atc_names'][x]
        )
        selected_atc_forecast = st.selectbox(
            "Kategori Obat (Forecasting):", 
            ias_metadata['modelable'], 
            format_func=lambda x: ias_metadata['atc_names'][x]
        )

    st.title("📊 Apotek Intelligence - Kelompok 2")
    st.subheader("Executive Dashboard: Intelligent Analytics System (IAS)")
    st.markdown("Sistem Intelijen Bisnis untuk Optimasi Inventori & Prediksi Permintaan.")

    # Hitung metrik KPI
    total_kritis = len(df_risk[df_risk['risk_label'] == 'KRITIS'])
    avg_mape = ias_metadata.get('avg_mape', 0)
    rentang_waktu = f"{ias_metadata['date_min'][:4]} - {ias_metadata['date_max'][:4]}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Kategori ATC", len(ias_metadata['atc_cols']))
    c2.metric("Akurasi Rata-rata", f"{100 - avg_mape:.1f}%", f"MAPE: {avg_mape}%", delta_color="inverse")
    c3.metric("Item Risiko Kritis", total_kritis)
    c4.metric("Rentang Historis", rentang_waktu)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📉 Analisis Historis", "📦 Pemetaan Risiko", "🔮 Proyeksi ARIMA"])

    with tab1:
        fig_trend = px.line(df_sales, x='datum', y=selected_atc_trend, 
                            title=f"Historis: {ias_metadata['atc_names'][selected_atc_trend]}",
                            template="simple_white")
        df_sales['MA30'] = df_sales[selected_atc_trend].rolling(30).mean()
        fig_trend.add_scatter(x=df_sales['datum'], y=df_sales['MA30'], mode='lines', name='MA 30 Hari')
        st.plotly_chart(fig_trend, width='stretch')

    with tab2:
        c_risk1, c_risk2 = st.columns([1, 2])
        c_risk1.dataframe(df_risk[['product', 'risk_label', 'movement_cat', 'volatility_cat']], height=400)
        
        color_map = {'KRITIS':'#e74c3c', 'TINGGI':'#e67e22', 'SEDANG':'#f1c40f', 'RENDAH':'#2ecc71', 'TIDAK DIKETAHUI':'#95a5a6'}
        fig_risk = px.scatter(df_risk, x='mean_daily_sales', y='cv', color='risk_label',
                              text='product', color_discrete_map=color_map,
                              title="Matriks Kuadran Inventori", template="simple_white")
        fig_risk.update_traces(textposition='top right')
        fig_risk.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig_risk.add_hline(y=0.5, line_dash="dash", line_color="orange")
        c_risk2.plotly_chart(fig_risk, width='stretch')

    with tab3:
        f_data = forecast_data[selected_atc_forecast]
        m1, m2, m3 = st.columns(3)
        m1.info(f"**Model:** ARIMA {f_data['order']}")
        m2.info(f"**MAE:** {f_data['mae']}")
        m3.info(f"**MAPE:** {f_data['mape']}%" if f_data['mape'] else "**MAPE:** N/A")
        
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=f_data['test_dates'], y=f_data['test_actual'], mode='lines', name='Aktual'))
        fig_fc.add_trace(go.Scatter(x=f_data['test_dates'], y=f_data['test_pred'], mode='lines', name='ARIMA', line=dict(dash='dash')))
        fig_fc.add_trace(go.Scatter(x=f_data['forecast_dates'], y=f_data['forecast_vals'], mode='lines+markers', name='Proyeksi 14 Hari'))
        
        fig_fc.update_layout(title=f"Proyeksi: {selected_atc_forecast}", template="simple_white")
        st.plotly_chart(fig_fc, width='stretch')

# ==========================================
# 7. MODUL 2: PKA (PHARMACY KNOWLEDGE ASSISTANT)
# ==========================================
elif pilihan_fitur == "💊 PKA (Asisten AI)":
    st.title("💊 Apotek Intelligence - Kelompok 2")
    st.subheader("Asisten AI: Pharmacy Knowledge Assistant (PKA)")
    st.caption("Prototipe RAG menggunakan TF-IDF & Llama-3 (Groq API)")
    
    vectorizer, tfidf_matrix = load_pka_models()
    chunks, pka_metadata = load_pka_data()
    
    if vectorizer is None or chunks is None:
        st.error("Data PKA tidak ditemukan. Pastikan file .pkl, .npz, pka_chunks.json, dan pka_metadata.json ada.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Konfigurasi PKA")
        groq_api_key = st.text_input("Groq API Key", type="password")
        st.caption(f"Knowledge Base: {pka_metadata.get('total_chunks', 0)} chunks")

    if "pka_messages" not in st.session_state:
        st.session_state.pka_messages = []

    for msg in st.session_state.pka_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"📚 Sumber: {', '.join(msg['sources'])}")

    if prompt := st.chat_input("Tanyakan seputar regulasi apotek..."):
        if not groq_api_key:
            st.warning("⚠️ Masukkan Groq API Key di sidebar!")
            st.stop()

        st.session_state.pka_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari konteks..."):
                retrieved_data, is_fallback = retrieve(
                    prompt, vectorizer, tfidf_matrix, chunks, 
                    pka_metadata.get('top_k', 3), pka_metadata.get('sim_threshold', 0.10)
                )

                if is_fallback:
                    jawaban = "Maaf, parameter pencarian gagal menemukan konteks di dokumen."
                    sumber = []
                else:
                    llm_prompt = build_prompt(prompt, retrieved_data)
                    jawaban = call_groq_api(llm_prompt, groq_api_key, pka_metadata.get('groq_model', 'llama3-70b-8192'))
                    sumber = list(dict.fromkeys(r['source'] for r in retrieved_data))

                st.markdown(jawaban)
                if sumber:
                    st.caption(f"📚 Sumber: {', '.join(sumber)}")
                
                with st.expander("🔍 Inspeksi Transparansi (Debug)"):
                    if not is_fallback:
                        for i, r in enumerate(retrieved_data, 1):
                            st.write(f"**[{i}] Skor: {r['score']} | {r['source']}**")
                            st.info(r['text'])

        st.session_state.pka_messages.append({"role": "assistant", "content": jawaban, "sources": sumber})