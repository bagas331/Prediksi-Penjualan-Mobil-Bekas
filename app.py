import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# ------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk antarmuka profesional tanpa emoji
st.markdown("""
<style>
    /* Styling Dasar */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header & Teks */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Kartu Metrik Kustom */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Kartu Informasi & Notifikasi */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        padding: 24px;
        border-radius: 6px;
        margin-top: 16px;
        margin-bottom: 24px;
    }
    
    .result-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .result-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    
    .result-range {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Konfigurasi plot global
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


# ------------------------------------------------------------
# FUNGSI PEMBANTU
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join('model', 'xgb_best_model.pkl')
    if not os.path.exists(model_path):
        st.error("File model tidak ditemukan. Jalankan pipeline data/model terlebih dahulu.")
        st.stop()
    return joblib.load(model_path)


@st.cache_resource
def load_preprocessor():
    prep_path = os.path.join('model', 'preprocessor.pkl')
    if not os.path.exists(prep_path):
        st.error("File preprocessor tidak ditemukan. Jalankan pipeline data/model terlebih dahulu.")
        st.stop()
    return joblib.load(prep_path)


@st.cache_data
def load_metadata():
    meta_path = os.path.join('model', 'model_metadata.json')
    if not os.path.exists(meta_path):
        st.error("File metadata tidak ditemukan. Jalankan pipeline data/model terlebih dahulu.")
        st.stop()
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_test_predictions():
    pred_path = os.path.join('model', 'test_predictions.pkl')
    if not os.path.exists(pred_path):
        return None
    return joblib.load(pred_path)


@st.cache_data
def load_feature_importance():
    fi_path = os.path.join('model', 'feature_importance.csv')
    if not os.path.exists(fi_path):
        return None
    return pd.read_csv(fi_path)


@st.cache_data
def load_dataset():
    csv_path = 'cleaned_used_car_dataset.csv'
    if not os.path.exists(csv_path):
        st.error("File dataset tidak ditemukan.")
        st.stop()
    return pd.read_csv(csv_path)


def format_rupiah(value):
    return f"Rp {value:,.0f}"


def calculate_vehicle_age(year):
    current_year = 2025
    return current_year - year


# ------------------------------------------------------------
# SIDEBAR INFO
# ------------------------------------------------------------
with st.sidebar.container():
    st.markdown("**Arsitektur Model**")
    st.caption("Algoritma: XGBoost Regressor")
    st.caption("Target: Selling Price (IDR)")
    st.caption("Status Model: Siap Digunakan")


# ------------------------------------------------------------
# HALAMAN: BERANDA
# ------------------------------------------------------------
def page_beranda():
    st.title("Sistem Estimasi Nilai Pasar Kendaraan")
    st.caption("Aplikasi Analitik dan Inferensi Machine Learning Berbasis Data Historis Transaksi")
    
    st.markdown("")
    
    # Ringkasan Model & Dataset
    try:
        metadata = load_metadata()
        metrics = metadata['metrics']

        st.subheader("Ringkasan Kinerja Model")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R2 Score", f"{metrics['r2']:.4f}", help="Koefisien Determinasi pada Data Uji")
        col2.metric("MAE", f"Rp {metrics['mae']:,.0f}", help="Mean Absolute Error")
        col3.metric("RMSE", f"Rp {metrics['rmse']:,.0f}", help="Root Mean Squared Error")
        col4.metric("MAPE", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error")

        st.markdown("")
        st.subheader("Informasi Data Pelatihan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Observasi", f"{metadata['dataset_size']:,} Baris")
        c2.metric("Dimensi Fitur Akhir", f"{metadata['n_features_after_encoding']} Kolom")
        c3.metric("Status Pipeline", "Tervalidasi")

    except Exception:
        st.warning("Metadata model belum tersedia. Silakan periksa direktori penyimpanan model.")




# HALAMAN: PREDIKSI HARGA
def page_prediksi():
    st.title("Inferensi Harga Jual Kendaraan")
    st.caption("Masukkan spesifikasi unit kendaraan untuk mendapatkan estimasi nilai pasar")

    try:
        model = load_model()
        preprocessor = load_preprocessor()
        metadata = load_metadata()
    except Exception as e:
        st.error(f"Gagal memuat komponen inferensi: {e}")
        st.stop()

    brand_model_map = metadata['brand_model_map']
    brands = sorted(brand_model_map.keys())

    # Muat dataset untuk referensi nilai dinamis berdasarkan model mobil
    df_dynamic = load_dataset()

    with st.container(border=True):
        st.markdown("##### Parameter Masukan")
        col1, col2, col3 = st.columns(3)

        with col1:
            brand = st.selectbox("Merek Kendaraan", options=brands)
            models_for_brand = sorted(brand_model_map.get(brand, []))
            car_model = st.selectbox("Model", options=models_for_brand)
            
            # Ambil nilai-nilai valid khusus untuk Brand dan Model ini dari dataset
            df_model = df_dynamic[(df_dynamic['Brand'] == brand) & (df_dynamic['Model'] == car_model)]
            available_fuels = sorted(df_model['Fuel_Type'].unique().tolist())
            available_transmissions = sorted(df_model['Transmission'].unique().tolist())
            available_seats = sorted(df_model['Seats'].unique().tolist())

            year = st.number_input(
                "Tahun Pembuatan",
                min_value=metadata['year_range'][0],
                max_value=metadata['year_range'][1],
                value=2020,
                step=1
            )
            seller_type = st.selectbox("Tipe Penjual", options=metadata['seller_types'])

        with col2:
            km_driven = st.number_input(
                "Jarak Tempuh Kumulatif (KM)",
                min_value=0,
                max_value=500000,
                value=50000,
                step=1000
            )
            owner = st.number_input(
                "Kepemilikan Ke-",
                min_value=metadata['owner_range'][0],
                max_value=metadata['owner_range'][1],
                value=1,
                step=1
            )
            fuel_type = st.selectbox("Bahan Bakar", options=available_fuels)
            transmission = st.selectbox("Transmisi", options=available_transmissions)

        with col3:
            max_power = st.number_input(
                "Tenaga Maksimum (BHP)",
                min_value=20.0,
                max_value=500.0,
                value=80.0,
                step=5.0,
                format="%.1f"
            )
            mileage = st.number_input(
                "Efisiensi Bahan Bakar (KM/L)",
                min_value=5.0,
                max_value=50.0,
                value=17.0,
                step=0.5,
                format="%.1f"
            )
            seats = st.selectbox("Kapasitas Penumpang (Kursi)", options=available_seats)

        submit_btn = st.button("Hitung Estimasi Harga", type="primary", use_container_width=True)

    if submit_btn:
        with st.spinner("Memproses inferensi..."):
            # Karena Engine CC selalu sama untuk setiap model di dataset ini, kita ambil otomatis
            df_for_cc = load_dataset()
            engine_cc = float(df_for_cc[(df_for_cc['Brand'] == brand) & (df_for_cc['Model'] == car_model)]['Engine_CC'].iloc[0])
            
            vehicle_age = calculate_vehicle_age(year)
            km_per_year = km_driven / max(vehicle_age, 1)
            power_per_cc = max_power / engine_cc

            input_data = pd.DataFrame({
                'Brand': [brand],
                'Model': [car_model],
                'Year': [year],
                'Fuel_Type': [fuel_type],
                'Transmission': [transmission],
                'KM_Driven': [km_driven],
                'Owner': [owner],
                'Engine_CC': [engine_cc],
                'Mileage_KMPL': [mileage],
                'Max_Power_BHP': [max_power],
                'Seats': [seats],
                'Seller_Type': [seller_type],
                'Vehicle_Age': [vehicle_age],
                'KM_Per_Year': [km_per_year],
                'Power_per_CC': [power_per_cc]
            })

            try:
                input_processed = preprocessor.transform(input_data)
                prediction = model.predict(input_processed)[0]

                margin = prediction * 0.10
                lower_bound = prediction - margin
                upper_bound = prediction + margin

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Estimasi Nilai Pasar Realistis</div>
                    <div class="result-value">{format_rupiah(prediction)}</div>
                    <div class="result-range">Rentang Estimasi Wajar (+/- 10%): <strong>{format_rupiah(lower_bound)}</strong> s/d <strong>{format_rupiah(upper_bound)}</strong></div>
                </div>
                """, unsafe_allow_html=True)

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Batas Bawah", format_rupiah(lower_bound))
                col_res2.metric("Nilai Tengah", format_rupiah(prediction))
                col_res3.metric("Batas Atas", format_rupiah(upper_bound))

                with st.expander("Tinjau Data Parameter yang Dikirimkan"):
                    st.dataframe(input_data.T.rename(columns={0: 'Nilai Input'}), use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kegagalan inferensi: {str(e)}")


# HALAMAN: EVALUASI MODEL
def page_evaluasi():
    st.title("Evaluasi Kinerja Model")
    st.caption("Pemeriksaan metrik performa dan validasi residual pada data pengujian (Test Set)")

    try:
        metadata = load_metadata()
        metrics = metadata['metrics']
    except Exception:
        st.error("Metadata model tidak tersedia.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2 Score", f"{metrics['r2']:.4f}")
    col2.metric("MAE", f"Rp {metrics['mae']:,.0f}")
    col3.metric("RMSE", f"Rp {metrics['rmse']:,.0f}")
    col4.metric("MAPE", f"{metrics['mape']:.2f}%")

    test_data = load_test_predictions()
    if test_data is not None:
        y_test = np.array(test_data['y_test'])
        y_pred = np.array(test_data['y_pred'])
        residuals = (y_test - y_pred) / 1e6

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            with st.container(border=True):
                st.markdown("##### Plot Aktual vs Prediksi")
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.scatter(y_test / 1e6, y_pred / 1e6, alpha=0.35, s=18, color='#2563eb', edgecolors='none')
                min_val = min(y_test.min(), y_pred.min()) / 1e6
                max_val = max(y_test.max(), y_pred.max()) / 1e6
                ax.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', lw=1.5, label='Ideal')
                ax.set_xlabel('Nilai Aktual (Juta IDR)', fontsize=9, color='#475569')
                ax.set_ylabel('Nilai Prediksi (Juta IDR)', fontsize=9, color='#475569')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(frameon=False, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with col_plot2:
            with st.container(border=True):
                st.markdown("##### Plot Sebaran Residual")
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.scatter(y_pred / 1e6, residuals, alpha=0.35, s=18, color='#0891b2', edgecolors='none')
                ax.axhline(y=0, color='#dc2626', linestyle='--', lw=1.5)
                ax.set_xlabel('Nilai Prediksi (Juta IDR)', fontsize=9, color='#475569')
                ax.set_ylabel('Residual (Juta IDR)', fontsize=9, color='#475569')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with st.container(border=True):
            st.markdown("##### Distribusi Kesalahan (Residuals)")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            sns.histplot(residuals, bins=45, kde=True, ax=ax, color='#475569', edgecolor=None)
            ax.axvline(x=0, color='#dc2626', linestyle='--', lw=1.5)
            ax.set_xlabel('Residual (Juta IDR)', fontsize=9, color='#475569')
            ax.set_ylabel('Frekuensi', fontsize=9, color='#475569')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    feat_imp = load_feature_importance()
    if feat_imp is not None:
        with st.container(border=True):
            st.markdown("##### Kepentingan Fitur (Feature Importance)")
            top_n = st.slider("Jumlah variabel:", min_value=5, max_value=min(25, len(feat_imp)), value=10)
            feat_imp_sorted = feat_imp.sort_values('Importance', ascending=True).tail(top_n)

            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
            ax.barh(feat_imp_sorted['Fitur'], feat_imp_sorted['Importance'], color='#334155', height=0.65)
            ax.set_xlabel('Tingkat Kepentingan', fontsize=9, color='#475569')
            ax.grid(True, linestyle=':', alpha=0.5, axis='x')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# HALAMAN: TENTANG DATASET
def page_dataset():
    st.title("Eksplorasi Basis Data")
    st.caption("Tinjauan struktur variabel, ringkasan statistika, dan sebaran agregat")

    df = load_dataset()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Observasi", f"{df.shape[0]:,}")
    c2.metric("Total Kolom", f"{df.shape[1]}")
    c3.metric("Data Kosong (Missing)", f"{df.isnull().sum().sum()}")

    st.markdown("")
    with st.container(border=True):
        st.markdown("##### Sampel Data Historis")
        st.dataframe(df.head(15), use_container_width=True)

    with st.container(border=True):
        st.markdown("##### Ringkasan Statistik Deskriptif")
        tab1, tab2 = st.tabs(["Variabel Numerik", "Variabel Kategorikal"])
        with tab1:
            st.dataframe(df.describe().T, use_container_width=True)
        with tab2:
            st.dataframe(df.describe(include='object').T, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("##### Distribusi Target (Selling Price)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Selling_Price_IDR'] / 1e6, bins=40, kde=True, ax=ax, color='#1e40af', edgecolor=None)
            ax.set_xlabel('Harga Jual (Juta IDR)', fontsize=9, color='#475569')
            ax.set_ylabel('Frekuensi', fontsize=9, color='#475569')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        with st.container(border=True):
            st.markdown("##### Sebaran Tipe Transmisi")
            fig, ax = plt.subplots(figsize=(6, 4))
            df['Transmission'].value_counts().plot(kind='bar', ax=ax, color='#475569', rot=0)
            ax.set_xlabel('Transmisi', fontsize=9, color='#475569')
            ax.set_ylabel('Jumlah Unit', fontsize=9, color='#475569')
            ax.grid(True, linestyle=':', alpha=0.6, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ------------------------------------------------------------
# ROUTING NAVIGATION
# ------------------------------------------------------------
pages = {
    "Menu Utama": [
        st.Page(page_beranda, title="Beranda", default=True),
        st.Page(page_prediksi, title="Prediksi Harga"),
        st.Page(page_evaluasi, title="Evaluasi Model"),
        st.Page(page_dataset, title="Tentang Dataset")
    ]
}

pg = st.navigation(pages)
pg.run()