import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def file_kosong(file):
    return (not os.path.exists(file)) or os.path.getsize(file) == 0

def init_csv(file, cols):
    if not os.path.exists(file):
        pd.DataFrame(columns=cols).to_csv(file, index=False)

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Hipertensi FKNN", 
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_LATIH = "data_latih.csv"
DATA_UJI = "data_uji.csv"
TETANGGA = "hasil_tetangga.csv"
KEANGGOTAAN = "hasil_keanggotaan_kelas.csv"
RINGKASAN = "hasil_ringkasan.csv"
DATA_NORM = "data_normalisasi.csv"

init_csv(DATA_LATIH, ["nama","tds","tdd","kelas"])
init_csv(DATA_UJI, ["tds","tdd","tanggal"])
init_csv(TETANGGA, ["tetangga_ke","tds_latih","tdd_latih","kelas","jarak"])
init_csv(KEANGGOTAAN, ["kelas","nilai_keanggotaan"])
init_csv(RINGKASAN, ["tds","tdd","k","m","kelas_hasil"])

# =====================================================
# PREPROCESSING (EDA & NORMALISASI)
# =====================================================
def normalisasi_data(df):
    df = df.copy()
    for f in ["tds", "tdd"]:
        if df[f].max() != df[f].min():
            df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())
        else:
            df[f] = 0
    return df

# =====================================================
# FKNN
# =====================================================
def fuzzy_membership_latih(kelas, kelas_list):
    return {k: 1.0 if k == kelas else 0.0 for k in kelas_list}

def fknn_predict(df, data_uji, k, m):
    fitur = ["tds", "tdd"]
    kelas_list = df["kelas"].unique().tolist()

    norm = df.copy()
    for f in fitur:
        norm[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())

    uji = {
        f: (data_uji[f] - df[f].min()) / (df[f].max() - df[f].min())
        for f in fitur
    }

    norm["jarak"] = np.sqrt(
        (norm["tds"] - uji["tds"])**2 +
        (norm["tdd"] - uji["tdd"])**2
    )

    tetangga = norm.sort_values("jarak").head(k)

    mu = {kls: 0.0 for kls in kelas_list}
    total = 0

    for _, r in tetangga.iterrows():
        d = r["jarak"] if r["jarak"] != 0 else 1e-6
        w = 1 / (d ** (2 / (m - 1)))
        total += w
        mu_latih = fuzzy_membership_latih(r["kelas"], kelas_list)
        for kls in kelas_list:
            mu[kls] += mu_latih[kls] * w

    for kls in mu:
        mu[kls] /= total

    return max(mu, key=mu.get), tetangga, mu

# =================== Styling Tampilan ===================
# CSS untuk styling aplikasi
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
        color: #E2E8F0;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #38B2AC 0%, #4299E1 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Card Styles */
    .card {
        background-color: #2D3748;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #38B2AC;
    }
    
    .card h2, .card h3 {
        color: #38B2AC;
        margin-top: 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(56, 178, 172, 0.3);
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background-color: #1A202C;
        padding: 1.5rem;
    }
    
    .sidebar .sidebar-header {
        color: #E2E8F0;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #4A5568;
    }
    
    /* Button Styles */
    .stButton button {
        background: linear-gradient(90deg, #38B2AC 0%, #4299E1 100%);
        color: white !important;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Input Styles */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #2D3748;
        color: #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem;
        border: 1px solid #4A5568;
    }
    
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #CBD5E0;
        font-weight: 600;
    }
    
    /* Dataframe Styles */
    .dataframe {
        background-color: #2D3748;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe thead th {
        background-color: #38B2AC;
        color: white;
        font-weight: 600;
    }
    
    .dataframe tbody tr:nth-child(odd) {
        background-color: rgba(45, 55, 72, 0.5);
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(56, 178, 172, 0.1);
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess {
        background-color: rgba(72, 187, 120, 0.2);
        border-left: 5px solid #48BB78;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: rgba(237, 137, 54, 0.2);
        border-left: 5px solid #ED8936;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stError {
        background-color: rgba(245, 101, 101, 0.2);
        border-left: 5px solid #F56565;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: rgba(66, 153, 225, 0.2);
        border-left: 5px solid #4299E1;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2D3748;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2D3748;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress .progress-bar {
        background-color: #38B2AC;
    }
    
    /* Metric Card */
    div[data-testid="metric-container"] {
        background-color: #2D3748;
        border: 1px solid #4A5568;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Radio Button */
    .stRadio > div {
        background-color: #2D3748;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4A5568;
    }
</style>
""", unsafe_allow_html=True)

# ================= Header =================
st.markdown("""
<div class="main-header">
    <h1>Sistem Klasifikasi Hipertensi</h1>
    <p>Metode Fuzzy K-Nearest Neighbor</p>
</div>
""", unsafe_allow_html=True)

# ================= UI =================
menu = st.sidebar.radio("Menu Navigasi", [
    "Data Latih",
    "Perhitungan FKNN",
    "Evaluasi Model"
])

if menu == "Data Latih":

    st.markdown("""
    <div class="card">
        <h2>Upload Data </h2>
        <p>Kolom wajib: nama, tds, tdd, kelas</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Unggah file data latih",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # ===== BACA FILE =====
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # ===== NORMALISASI NAMA KOLOM =====
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
            )

            # ===== VALIDASI KOLOM =====
            required_cols = {"nama", "tds", "tdd", "kelas"}
            if not required_cols.issubset(df.columns):
                st.error(
                    f"Kolom tidak sesuai.\n"
                    f"Ditemukan: {list(df.columns)}\n"
                    f"Wajib: {required_cols}"
                )
                st.stop()

            # ===== SIMPAN =====
            df = df[list(required_cols)]
            df.to_csv(DATA_LATIH, index=False)

            st.success(f"✅ {len(df)} data latih berhasil dimuat.")
            
            st.markdown("""
            <div class="card">
                <h3>Data Latih</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    else:
        if os.path.exists(DATA_LATIH) and os.path.getsize(DATA_LATIH) > 0:
            df = pd.read_csv(DATA_LATIH)
            st.markdown("""
            <div class="card">
                <h3>Data Latih</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Belum ada data latih.")


elif menu == "Perhitungan FKNN":

    # ================= BACA DATA LATIH =================
    df = pd.read_csv(DATA_LATIH)
    if df.empty:
        st.warning("Data latih kosong")
        st.stop()

    st.markdown("""
    <div class="card">
        <h2>Input Data Uji</h2>
        <p>Masukkan data pasien untuk diklasifikasi</p>
    </div>
    """, unsafe_allow_html=True)

    # ================= INPUT PARAMETER =================
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        nama_uji = st.text_input("Nama Pasien Uji", placeholder="Masukkan nama pasien")
    with c1:
        tds = st.number_input("TDS Uji", min_value=0, max_value=300, value=120)
    with c2:
        tdd = st.number_input("TDD Uji", min_value=0, max_value=200, value=80)
    with c3:
        k = st.number_input("K", min_value=1, max_value=len(df), value=5)

    m = st.number_input("m (pangkat fuzzy)", value=2)

    # ================= PROSES =================
    if st.button("Hitung FKNN"):

        if nama_uji.strip() == "":
            st.warning("Nama pasien uji tidak boleh kosong.")
            st.stop()

        # ---------- Simpan Data Uji ----------
        data_uji = pd.DataFrame([{
            "nama": nama_uji,
            "tds": tds,
            "tdd": tdd
        }])

        data_uji.to_csv(
            DATA_UJI,
            mode="a",
            header=file_kosong(DATA_UJI),
            index=False
        )

        # ---------- Prediksi FKNN ----------
        hasil, tetangga, mu = fknn_predict(
            df,
            {"tds": tds, "tdd": tdd},
            k,
            m
        )

        # ================= NORMALISASI (UNTUK TABEL SAJA) =================
        tds_min, tds_max = df["tds"].min(), df["tds"].max()
        tdd_min, tdd_max = df["tdd"].min(), df["tdd"].max()

        # ---------- Simpan Tetangga (DENGAN NORM) ----------
        tetangga_df = pd.DataFrame([
            {
                "nama_uji": nama_uji,
                "tetangga_ke": i + 1,
                "nama_latih": row["nama"],

                # ===== DATA ASLI =====
                "tds": row["tds"],
                "tdd": row["tdd"],

                # ===== DATA NORMALISASI =====
                "tds_norm": (row["tds"] - tds_min) / (tds_max - tds_min)
                            if tds_max != tds_min else 0,
                "tdd_norm": (row["tdd"] - tdd_min) / (tdd_max - tdd_min)
                            if tdd_max != tdd_min else 0,

                "kelas": row["kelas"],
                "jarak": row["jarak"]
            }
            for i, row in tetangga.iterrows()
        ])

        tetangga_df.to_csv(
            TETANGGA,
            mode="a",
            header=file_kosong(TETANGGA),
            index=False
        )

        # ---------- Simpan Keanggotaan ----------
        mu_df = pd.DataFrame([
            {
                "nama_uji": nama_uji,
                "kelas": kls,
                "nilai_keanggotaan": val
            }
            for kls, val in mu.items()
        ])

        mu_df.to_csv(
            KEANGGOTAAN,
            mode="a",
            header=file_kosong(KEANGGOTAAN),
            index=False
        )

        # ---------- Simpan Ringkasan ----------
        ringkasan_df = pd.DataFrame([{
            "nama_uji": nama_uji,
            "tds": tds,
            "tdd": tdd,
            "k": k,
            "m": m,
            "kelas": hasil
        }])

        ringkasan_df.to_csv(
            RINGKASAN,
            mode="a",
            header=file_kosong(RINGKASAN),
            index=False
        )

        # ================= OUTPUT =================
        st.markdown(f"""
        <div class="card">
            <h2>Hasil Klasifikasi</h2>
            <p>Pasien <strong>{nama_uji}</strong> diklasifikasikan sebagai: <strong>{hasil}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>Nilai Keanggotaan Fuzzy</h3>
        </div>
        """, unsafe_allow_html=True)
        st.bar_chart(mu_df.set_index("kelas"))

        st.markdown("""
        <div class="card">
            <h3>Tetangga Terdekat (dengan Normalisasi)</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(tetangga_df, use_container_width=True)

    # ================= RIWAYAT DATA =================
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.expander("Riwayat Data Uji"):
        if os.path.exists(DATA_UJI) and os.path.getsize(DATA_UJI) > 0:
            st.dataframe(pd.read_csv(DATA_UJI), use_container_width=True)
        else:
            st.info("Belum ada data uji.")

    with st.expander("Riwayat Tetangga"):
        if os.path.exists(TETANGGA) and os.path.getsize(TETANGGA) > 0:
            st.dataframe(pd.read_csv(TETANGGA), use_container_width=True)
        else:
            st.info("Belum ada data tetangga.")

    with st.expander("Riwayat Keanggotaan"):
        if os.path.exists(KEANGGOTAAN) and os.path.getsize(KEANGGOTAAN) > 0:
            st.dataframe(pd.read_csv(KEANGGOTAAN), use_container_width=True)
        else:
            st.info("Belum ada data keanggotaan.")

    with st.expander("Riwayat Hasil Klasifikasi"):
        if os.path.exists(RINGKASAN) and os.path.getsize(RINGKASAN) > 0:
            st.dataframe(pd.read_csv(RINGKASAN), use_container_width=True)
        else:
            st.info("Belum ada hasil klasifikasi.")

#===========================
elif menu == "Evaluasi Model":
    df = pd.read_csv(DATA_LATIH)
    if df.empty:
        st.warning("Data latih kosong")
        st.stop()

    st.markdown("""
    <div class="card">
        <h2>Evaluasi Model FKNN</h2>
        <p>Upload data uji untuk mengevaluasi performa model</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload data uji (tds, tdd, kelas [opsional])",
        type=["csv", "xlsx"]
    )

    c1, c2 = st.columns(2)
    with c1:
        k = st.number_input("K", 1, len(df), 5)
    with c2:
        m = st.number_input("m", value=2)

    if uploaded and st.button("Jalankan Evaluasi FKNN"):

        # ===== BACA FILE CSV / EXCEL =====
        if uploaded.name.endswith(".csv"):
            df_uji = pd.read_csv(uploaded)
            if len(df_uji.columns) == 1:
                df_uji = pd.read_csv(uploaded, sep=";")
        else:
            df_uji = pd.read_excel(uploaded)

        # ===== NORMALISASI KOLOM =====
        df_uji.columns = df_uji.columns.str.lower().str.strip()

        # ===== VALIDASI KOLOM WAJIB =====
        if not {"tds", "tdd"}.issubset(df_uji.columns):
            st.error("File harus memiliki kolom: tds dan tdd")
            st.stop()

        y_true, y_pred = [], []

        # ===== PREDIKSI =====
        for _, row in df_uji.iterrows():
            hasil, _, _ = fknn_predict(
                df,
                {"tds": row["tds"], "tdd": row["tdd"]},
                k,
                m
            )
            y_pred.append(hasil)
            if "kelas" in df_uji.columns:
                y_true.append(row["kelas"])

        df_uji["kelas_prediksi"] = y_pred

        st.markdown("""
        <div class="card">
            <h3>Hasil Prediksi</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_uji, use_container_width=True)

        # ===== EVALUASI =====
        if "kelas" in df_uji.columns:
            acc = (df_uji["kelas"] == df_uji["kelas_prediksi"]).mean() * 100
            
            st.markdown(f"""
            <div class="card">
                <h3>Metrik Evaluasi</h3>
                <p><strong>Akurasi:</strong> {acc:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            report = classification_report(y_true, y_pred, output_dict=True)
            st.markdown("""
            <div class="card">
                <h3>Classification Report</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.info("Kolom 'kelas' tidak tersedia. Akurasi tidak dihitung.")