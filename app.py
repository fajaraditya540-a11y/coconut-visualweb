import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import ee
import folium
from streamlit_folium import st_folium
import json 
from google.oauth2 import service_account 

# ==============================================================================
# ALUR PROSES APLIKASI (PROJECT COCONUT VISUALIZER)
# ==============================================================================
# Script ini adalah Web-GIS & Simulator Tekno-Ekonomi terintegrasi yang 
# berjalan melalui 5 fase komputasi utama:
#
# 1. SETUP & INISIALISASI (Environment & UI Config)
#    - Import library esensial: UI (Streamlit), Data (Pandas/Numpy), ML (Sklearn), & Geospasial (Earth Engine, Folium).
#    - Konfigurasi antarmuka web (Wide layout, Custom CSS, Font Inter Tight).
#    - Autentikasi dan inisiasi koneksi ke server Google Earth Engine secara otomatis.
#
# 2. STATE MANAGEMENT & INTERFACE PANEL (Sidebar)
#    - Menggunakan 'st.session_state' untuk mengunci dan menyimpan variabel simulasi secara konsisten.
#    - Membangun Sidebar yang berisi Slider parameter (Volume Limbah, Efisiensi, Harga Karbon, CapEx, OpEx).
#    - Menyediakan fungsi fungsi override (Skenario Ideal vs Kritis) untuk demonstrasi instan saat sidang.
#
# 3. ENGINE KOMPUTASI INTI (Model Geokimia & Finansial)
#    - Menghitung serapan CO2 aktual berdasarkan rumus stoikiometri (kadar MgO slag & efisiensi).
#    - Menghitung dosis rasio aplikasi limbah terhadap lahan (Ton/Ha) dan prediksi nilai pH ekuilibrium.
#    - Mengalkulasi metrik finansial (Revenue Kredit Karbon, Listrik TCES, dan Avoided Cost Industri).
#    - Logika kondisional evaluasi ekonomi: Menentukan apakah proyek Profitable (Payback) atau Unfeasible (Rugi).
#
# 4. RENDERING DASHBOARD STATIS (Tab 0, 1, & 2)
#    - TAB 0 (Executive Summary): Menampilkan narasi latar belakang (Paradoks Hijau vs Inovasi), metrik ESG, 
#      dan grafik Matplotlib dinamis (Komparasi LCOE & Neraca Emisi).
#    - TAB 1 (Techno-Economic): Menampilkan scorecard status lahan, metrik kelayakan investasi, dan tabel arus kas.
#    - TAB 2 (Korelasi Spasial): Menyajikan analisis geokomputasi dari 5 indeks satelit asli (LST, FVC, NDWI, NDMI, SBI) 
#      vs Elevasi DEMNAS untuk membuktikan secara fisis anomali dan degradasi lingkungan di lapangan.
#
# 5. KOMPUTASI SPASIAL DINAMIS & WEB-GIS (Tab 3)
#    - Menentukan Region of Interest (ROI) dengan buffer 10 Km di pusat Kawasi, Pulau Obi.
#    - Pre-Processing GEE: Melakukan Cloud Masking menggunakan algoritma QA60 bitwise pada Sentinel-2.
#    - Analisis Topografi & Hidrologi (DEMNAS 8m): Mengekstrak nilai lereng (slope) dan buffer jarak aman dari badan air (> 200m).
#    - Multi-Criteria Analysis: Menggabungkan (Boolean Logic) kriteria lereng aman dan jarak air menjadi Masking Zona ERW.
#    - Ekstraksi Area (reduceRegion): Menghitung luas (Hektar) piksel valid secara langsung dari server Google Earth Engine.
#    - Sistem Notifikasi Validasi: Membandingkan kecukupan daya dukung lahan spasial dengan kebutuhan dosis volume limbah.
#    - Rendering Peta: Memuat seluruh layer analisis (Hillshade, NDVI, Bahaya Lereng, Badan Air, Zona Rekomendasi) ke dalam peta Folium interaktif.
# ==============================================================================

# ==========================================
# SETUP HALAMAN MUKA
# ==========================================
st.set_page_config(page_title="COCONUT Simulator", page_icon="🥥", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"], * {
            font-family: 'Inter Tight', sans-serif !important;
        }
        .big-number { font-size: 28px; font-weight: 700; color: #2e8b57; margin-bottom: 0; }
        .big-label { font-size: 14px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 1px;}
        .esg-box { padding: 15px; border-radius: 10px; background: white; border-left: 5px solid #2e8b57; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# INISIALISASI GEE & ML MODEL
# ==========================================

@st.cache_resource
def init_gee():
    try:
        # 1. Tarik data JSON dari brankas Streamlit Secrets
        key_dict = json.loads(st.secrets["EE_KEYS"])
        
        # 2. Buat Kredensial pakai metode modern Google OAuth2
        credentials = service_account.Credentials.from_service_account_info(key_dict)
        
        # 3. Inisialisasi GEE dengan kredensial tersebut
        ee.Initialize(credentials, project='obi-project-495300')
        print("GEE Berhasil Diinisialisasi di Cloud!")
        
    except Exception as e:
        # Jika gagal, tampilkan pesan aslinya di layar aplikasi dan STOP kodingan
        st.error(f"🚨 GAGAL INISIALISASI GEE. Pesan Error Asli: {e}")
        st.stop() # Menghentikan script agar tidak lanjut ke baris 461

init_gee()

def add_ee_layer(m, ee_image_object, vis_params, name, opacity=1.0):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True,
        opacity=opacity # Parameter ini yang tadi dicari oleh Python!
    ).add_to(m)

# ==========================================
# STATE MANAGEMENT (SKENARIO INTERAKTIF)
# ==========================================
if 'vol_limbah' not in st.session_state: st.session_state.vol_limbah = 2000
if 'eff' not in st.session_state: st.session_state.eff = 80
if 'harga_c' not in st.session_state: st.session_state.harga_c = 58800
if 'capex' not in st.session_state: st.session_state.capex = 10.0
if 'opex' not in st.session_state: st.session_state.opex = 15000

def set_scenario_ideal():
    st.session_state.vol_limbah = 15000
    st.session_state.eff = 95
    st.session_state.harga_c = 85000
    st.session_state.capex = 25.0

def set_scenario_kritis():
    st.session_state.vol_limbah = 25000
    st.session_state.eff = 60
    st.session_state.capex = 10.0

# ==========================================
# HEADER & STRUKTUR TAB
# ==========================================
st.markdown("""
    <div style="background: linear-gradient(135deg, #0B3D0B 0%, #2e8b57 100%); padding: 30px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 10px 20px rgba(46, 139, 87, 0.2);">
        <h1 style="margin: 0; font-size: 36px; font-weight: 700; letter-spacing: -1px;">🥥 Project COCONUT Visualizer</h1>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 18px;">Integrated Carbon Hydro-Geology & Techno-Economic Framework | Pulau Obi</p>
    </div>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3 = st.tabs([" Executive Summary", " Techno-Economic Panel", " Analisis Korelasi Spasial", " Interactive Spatial Map"])

# ==========================================
# SIDEBAR DENGAN SESSION STATE
# ==========================================
with st.sidebar:
    st.header("Panel Simulator")
    volume_limbah = st.slider('Volume Limbah (Ton):', 1000, 25000, key='vol_limbah', step=500)
    efisiensi = st.slider('Efisiensi Karbonatasi (%):', 10, 100, key='eff', step=5)
    harga_karbon = st.slider('Harga Karbon (Rp/Ton):', 20000, 150000, key='harga_c', step=1000)
    capex_miliar = st.slider('CapEx (Miliar Rp):', 1.0, 100.0, key='capex', step=1.0)
    opex_per_ton = st.slider('OpEx Logistik (Rp/Ton):', 5000, 100000, key='opex', step=5000)

# ==========================================
# LOGIKA KOMPUTASI (FINAL MODEL)
# ==========================================
kadar_mgo_slag = 0.2751
batas_lahan_tersedia = 150.0 

serapan_real = volume_limbah * kadar_mgo_slag * (44.0 / 40.3) * (efisiensi / 100)
dosis_aplikasi_ha = volume_limbah / batas_lahan_tersedia

revenue_karbon = serapan_real * harga_karbon
konstanta_listrik_rp_per_ton = 53.33 * 700 
revenue_fisik = volume_limbah * konstanta_listrik_rp_per_ton 
avoided_cost = volume_limbah * 73000

total_revenue = revenue_karbon + revenue_fisik + avoided_cost
total_opex = volume_limbah * opex_per_ton
laba_bersih = total_revenue - total_opex

import math
prediksi_ph = 7.0 - (2.5 * math.exp(-0.2 * dosis_aplikasi_ha))

# ==========================================
# TAB 0: EXECUTIVE SUMMARY & VISUALIZER INTRO
# ==========================================
with tab0:
   
    st.markdown("""
<div style="display: flex; gap: 20px; margin-top: 15px; margin-bottom: 20px;">
<div style="flex: 1; background-color: #fffaf0; padding: 20px; border-radius: 10px; border-top: 5px solid #e74c3c; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
<h3 style="color: #c0392b; margin-top: 0; font-size: 18px;"> Ironi Transisi Energi </h3>
<p style="font-size: 14px; text-align: justify; color: #333; line-height: 1.6;">
Transisi ke kendaraan listrik sangat membutuhkan nikel. Namun di Pulau Obi, industri ini memicu <b>"Paradoks Hijau"</b>:
</p>
<ul style="font-size: 14px; color: #333; padding-left: 20px; line-height: 1.6;">
<li>Produksi menghasilkan jutaan ton limbah cair beracun (tailing) dan emisi udara masif.</li>
<li>Warga lokal sempat hidup dalam krisis listrik di tengah gemerlap kawasan industri yang haus energi.</li>
</ul>
</div>
<div style="flex: 1; background-color: #f0fdf4; padding: 20px; border-radius: 10px; border-top: 5px solid #2ecc71; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
<h3 style="color: #27ae60; margin-top: 0; font-size: 18px;"> Inovasi COCONUT </h3>
<p style="font-size: 14px; text-align: justify; color: #333; line-height: 1.6;">
Kami mengintegrasikan penangkapan emisi PLTU captive dengan mereaksikan limbah Tailing HPAL dan Slag RKEF melalui metode Karbonatasi Mineral. Proses ini mengonversi eksternalitas industri menjadi dua komoditas bernilai tinggi:  
</p>
<ul style="font-size: 14px; color: #333; padding-left: 20px; line-height: 1.6;">
<li><b>Baterai Termal (TCES):</b> Pemanfaatan mineral Magnesit (MgCO3) untuk menyimpan limbah panas smelter tanpa penyusutan energi (Zero Thermal Loss). Energi yang tersimpan dialirkan kembali melalui turbin ORC untuk menyuplai listrik baseload secara konstan bagi masyarakat. </li>
<li><b>Pupuk ERW (TCLP Safe):</b> Pemurnian residu silika yang tersertifikasi aman dari logam berat karsinogenik. Residu ini diaplikasikan langsung pada lahan pertanian warga sebagai agen pemulih hara sekaligus penangkap CO2 atmosferik secara permanen.  </li>
</ul>
</div>
</div>
<div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #2980b9; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;">
<h3 style="color: #2980b9; margin-top: 0; font-size: 18px;"> Dampak Makro & Visi Proyek</h3>
<p style="font-size: 14px; margin-bottom: 0; color: #333; line-height: 1.6;">
Memusnahkan ancaman limbah beracun, memutus krisis energi dengan pasokan listrik mandiri bagi Desa Kawasi, merestorasi lahan agrikultur yang terdegradasi, dan <b>menjadikan limbah industri itu sendiri sebagai senjata perisai melawan pemanasan global</b>.
</p>
</div>
""", unsafe_allow_html=True)

    colA, colB = st.columns([1.5, 1])

    with colA:
        st.subheader("Integrasi Geologi Komputasi")
        st.markdown("""
        Simulator ini dibangun menggunakan tiga pilar komputasi utama untuk memastikan akurasi dan skalabilitas proyek:
        * **Geospasial & Analisis Korelasi (Earth Engine API):** Pemrosesan satelit berbasis cloud (Sentinel-2 & DEMNAS 8m) untuk penentuan zona aman ERW serta ekstraksi 5 indeks parameter (LST, FVC, NDWI, NDMI, SBI) guna memetakan anomali termal industri dan degradasi lahan secara spasial.
        * **Tekno-Ekonomi:** Integrasi data fisik TCES (Thermal Energy Storage) ke dalam proyeksi arus kas Multi-Revenue Stream (Kredit Karbon, Elektrifikasi Desa, dan Avoided Cost Smelter) untuk menguji kelayakan finansial secara riil berdasarkan parameter input simulator.
        """)
        
        st.subheader("Intervensi Cepat (Skenario)")
        st.markdown("Klik tombol di bawah untuk mendemonstrasikan pergeseran algoritma simulasi:")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.button("Load Skenario Ideal (Skalasi Tinggi)", on_click=set_scenario_ideal, use_container_width=True)
        with btn_col2:
            st.button("Load Skenario Kritis (Anomali Input)", on_click=set_scenario_kritis, use_container_width=True)

    with colB:
        st.subheader("Dampak Makro (ESG)")
        st.markdown('<div class="esg-box">🌍 <b>Environmental:</b> <span style="color:#7f8c8d;">Penyerapan 10.000 Ton CO2/Tahun & reklamasi 5.500 Ha lahan kritis.</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="esg-box">👥 <b>Social:</b> <span style="color:#7f8c8d;">Elektrifikasi baseload 1 MW untuk Desa Kawasi (Menggantikan Diesel).</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="esg-box">🏛️ <b>Governance:</b> <span style="color:#7f8c8d;">Integritas kredit karbon dan kepatuhan absolut standar limbah B3 (TCLP).</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Komparasi Kinerja Proyek (Sebelum vs Sesudah COCONUT)")
    
    # Grafik Komparasi Sederhana
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: LCOE
    ax1.bar(["Genset Diesel\n(Existing)", "Microgrid COCONUT\n(New)"], [3000, 687], color=['#e74c3c', '#2ecc71'])
    ax1.set_title("Biaya Pokok Penyediaan Listrik (LCOE) - Rp/kWh", fontweight='bold')
    ax1.set_ylabel("Rupiah / kWh")
    for i, v in enumerate([3000, 687]): ax1.text(i, v + 50, f"Rp {v}", ha='center', fontweight='bold')

    # Chart 2: Net Emisi
    ax2.bar(["Tanpa Intervensi", "Dengan ERW COCONUT"], [10000, -10000], color=['#95a5a6', '#2ecc71'])
    ax2.set_title("Neraca Emisi Karbon (Ton CO2e/Tahun)", fontweight='bold')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel("Ton CO2")
    
    st.pyplot(fig_comp)

# ==========================================
# TAB 1: TECHNO-ECONOMIC PANEL
# ==========================================

    st.markdown("###  Evaluasi Tekno-Ekonomi & Arus Kas Proyek sirkular")

with st.expander("Panduan Finansial & Multi-Revenue Stream (Klik)", expanded=True):
    st.markdown("""
    Panel ini mengintegrasikan **Variabel Fisika Sistem** (Kapasitas Thermal Energy Storage/TCES dan Volume Olahan Limbah) ke dalam proyeksi finansial riil. Simulator menguji kelayakan ekonomi investasi jangka panjang melalui skenario pendapatan majemuk.
    
    Berikut adalah breakdown dari mekanisme **Multi-Revenue Stream** proyek COCONUT:
    """)
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("""
        * **1. Carbon Credit (Kredit Karbon):**
            * *Sumber:* Diperoleh dari netralisasi CO₂ aktual melalui proses pelapukan batuan buatan (*Enhanced Rock Weathering* / ERW).
            * *Mekanisme:* Setiap ton CO₂ yang berhasil dikunci oleh kalsium/magnesium silika secara permanen akan dikonversi menjadi unit kredit karbon bersertifikasi untuk dijual ke pasar sukarela (*Voluntary Carbon Market*).
        * **2. Desa Elektrifikasi (Rural Electrification):**
            * *Sumber:* Pemanfaatan energi panas buang (*waste heat*) smelter yang ditangkap oleh **Baterai Termal TCES**.
            * *Mekanisme:* Panas yang disimpan diubah kembali menjadi energi listrik stabil (*baseload*) untuk dijual ke masyarakat lingkar tambang/PLN melalui skema mikro-grid, menggantikan diesel/PLTD.
        """)
        
    with m_col2:
        st.markdown("""
        * **3. Avoided Cost Smelter (Penghematan Industri):**
            * *Sumber:* Insentif dari pengelolaan limbah slag dan tailing langsung di sumbernya.
            * *Mekanisme:* Nilai keekonomian yang lahir karena perusahaan smelter tidak perlu lagi membayar biaya penimbunan logistik, pajak penempatan limbah B3, atau denda penanganan lingkungan (*tipping fee* terhindarkan).
        * **4. Metrik Kelayakan Investasi:**
            * *NPV (Net Present Value):* Menghitung nilai bersih proyek saat ini. Jika NPV > 0, proyek ini menghasilkan profit di atas biaya modal.
            * *IRR (Internal Rate of Return):* Tingkat pengembalian modal internal. Harus lebih tinggi dari suku bunga pinjaman bank agar menarik di mata investor.
            * *Payback Period:* Durasi waktu yang dibutuhkan proyek untuk mengembalikan seluruh modal awal (CapEx).
        """)
    
    st.info("💡 **Sensitivitas Nilai:** Kamu bisa menggeser parameter **CapEx, OpEx, dan Harga Karbon** di *sidebar* untuk melihat seberapa tangguh proyek ini dalam menghadapi fluktuasi pasar komoditas global secara dinamis.")

# --- MASUK KE KODE FINANSIAL / TABEL / GRAFIK FINANSIAL ---
with tab1:
    status_lahan, warna_lahan = f"TCLP Safe ({dosis_aplikasi_ha:.1f} Ton/Ha)", "#2ecc71"
    if laba_bersih > 0:
        payback_thn = (capex_miliar * 1e9) / laba_bersih
        status_ekonomi, warna_ekonomi, pesan_ekonomi = f"PROFITABLE ({payback_thn:.1f} Thn Payback)", "#2ecc71", f"Rp {laba_bersih:,.0f} / Thn"
    else:
        status_ekonomi, warna_ekonomi, pesan_ekonomi = "UNFEASIBLE (RUGI)", "#e74c3c", f"Rp {laba_bersih:,.0f} / Thn"

    st.markdown(f"""
    <div style="display: flex; gap: 20px;">
        <div style="background: white; padding: 20px; border-radius: 12px; flex: 1; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 5px solid #2e8b57;">
            <p class="big-label">🌍 SPASIAL & GEOKIMIA</p>
            <p class="big-number">{serapan_real:,.1f} Ton/Thn</p>
            <p style="margin: 5px 0 15px 0; color: #7f8c8d;">Serapan CO2 Efektif</p>
            <span style="background:{warna_lahan}; color:white; padding:5px 12px; border-radius:4px; font-size:13px; font-weight:600;">{status_lahan}</span>
        </div>
        <div style="background: white; padding: 20px; border-radius: 12px; flex: 1; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 5px solid {warna_ekonomi};">
            <p class="big-label">💰 KELAYAKAN FINANSIAL</p>
            <p class="big-number" style="color: {warna_ekonomi};">{pesan_ekonomi}</p>
            <p style="margin: 5px 0 15px 0; color: #7f8c8d;">Estimasi Laba Bersih</p>
            <span style="background:{warna_ekonomi}; color:white; padding:5px 12px; border-radius:4px; font-size:13px; font-weight:600;">{status_ekonomi}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💰 Detail Arus Kas Finansial (Multi-Revenue)")
    df_ekonomi = pd.DataFrame({
        "Komponen Finansial": ["1. Penjualan Kredit Karbon", "2. Energi Listrik (TCES)", "3. Avoided Cost (Smelter)", "Total Biaya Operasional (OpEx)", "Laba Bersih Tahunan"],
        "Nilai Aktual": [f"Rp {revenue_karbon:,.0f}", f"Rp {revenue_fisik:,.0f}", f"Rp {avoided_cost:,.0f}", f"- Rp {total_opex:,.0f}", f"Rp {laba_bersih:,.0f}"]
    })
    st.dataframe(df_ekonomi, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: GRAFIK & TABEL GEOKIMIA
# ==========================================
with tab2:
    st.subheader("Ringkasan Parameter & Output Fisik Sistem")
    df_serapan = pd.DataFrame({
        "Parameter Komputasi": ["Volume Tailing & Slag", "Target Serapan Efektif", "Prediksi pH Tanah Akhir"],
        "Nilai Aktual": [f"{volume_limbah:,.0f} Ton", f"{serapan_real:,.1f} Ton", f"{prediksi_ph:.2f}"]
    })
    st.dataframe(df_serapan, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- GRAFIK KORELASI LST ---
    st.subheader("Analisis Korelasi Spasial: Topografi vs Suhu Permukaan (LST)")
    
    col_text, col_plot = st.columns([1, 1.5])

    with col_text:
        st.markdown("### 🏜️ Elevasi vs LST (Termal)")
        st.markdown("""
        Grafik ini menguji korelasi antara **Ketinggian (DEMNAS 8m)** dengan **Suhu Permukaan (LST)** dari satelit untuk mendeteksi penyimpangan iklim mikro.
        
        **Poin Kunci:**
        1. **Garis Tren Alami:** Secara umum, semakin tinggi elevasi menuju pedalaman bukit Obi, suhu permukaan akan semakin sejuk mengikuti hukum meteorologi dasar.
        2. **Anomali Smelter:** Terjadi deviasi ekstrem pada elevasi rendah (0 - 50 mdpl). Suhu melonjak sangat tinggi mencapai 34°C - 39°C. Ini adalah bukti riil emisi panas buang (*waste heat*) yang dilepaskan oleh aktivitas industri smelter di pesisir Kawasi.
         """)
        st.info("**Apa Maksudnya?:**\n\nZona anomali segitiga merah inilah yang menjadi target intervensi sistem **TCES (Baterai Termal)** COCONUT untuk menangkap dan mengunci panas terbuang tersebut menjadi pasokan listrik baseload desa.")

    with col_plot:
        # Menampilkan gambar statis hasil ekstraksi GEE dari Colab
        st.image("grafik_lst.png", caption="Data Spasial Aktual: Landsat 8/9 & DEMNAS 8m via Google Earth Engine", use_container_width=True)

    # -------------------------------------------------------
    # GRAFIK KEDUA (ELEVASI VS FVC)
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("Analisis Korelasi Spasial: Topografi vs Kerapatan Kanopi Hutan (FVC)")
    st.markdown("Memetakan klaster bukaan lahan tambang aktif dan degradasi vegetasi berdasarkan zonasi ketinggian.")

    # Membuat dua kolom baru (Row 2)
    col_text2, col_plot2 = st.columns([1, 1.5])

    with col_text2:
        st.markdown("### 🌿 Elevasi vs FVC (Vegetasi)")
        st.markdown("""
        Grafik ini memetakan fraksi tutupan kanopi secara aktual dari satelit Sentinel-2 terhadap zonasi ketinggian DEMNAS 8m untuk melacak jejak pembukaan lahan.
        
        **Interpretasi Geokomputasi:**
        * **Profil Hutan Alami (Lingkaran Hijau):** Secara natural, Pulau Obi memiliki tutupan kanopi yang rapat (FVC > 0.4 hingga 1.0) yang tersebar merata dari elevasi pesisir hingga puncak perbukitan (>300 mdpl).
        * **Anomali Deforestasi Masif (Kotak Oranye):** Terlihat penumpukan data ekstrem (*vertical clustering*) tepat pada nilai FVC = 0 (gundul total) yang membentang dari elevasi 0 hingga 350 mdpl. Pola vertikal ini membuktikan bahwa pengupasan lahan terjadi di seluruh gradien topografi; mulai dari tapak smelter dan pelabuhan di pesisir, hingga pengupasan *topsoil* untuk *open-pit* di dataran tinggi.
        """)
        st.info("**Apa Maksudnya?:**\n\nArea dengan FVC di bawah ambang batas kritis (< 0.35) yang kehilangan daya ikat tanah inilah yang menjadi target prioritas penaburan produk *Soil Ameliorant* dari sistem COCONUT untuk mencegah erosi laterit masif dan memicu revegetasi cepat.")

    with col_plot2:
        st.image("grafik_fvc.png", caption="Data Spasial Aktual: Sentinel-2 (10m) & DEMNAS 8m via Google Earth Engine", use_container_width=True)

    # -------------------------------------------------------
    # GRAFIK KETIGA (ELEVASI VS NDWI)
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("Analisis Korelasi Spasial: Topografi vs Kelembaban Lahan (NDWI)")
    st.markdown("Mengukur potensi hidrologi permukaan tanah untuk mengoptimalkan reaksi pelapukan buatan (ERW).")

    # Membuat dua kolom baru (Row 3)
    col_text3, col_plot3 = st.columns([1, 1.5])

    with col_text3:
        st.markdown("### 🌧️ Elevasi vs NDWI (Moisture)")
        st.markdown("""
        Grafik aktual ini memetakan retensi air permukaan dan keberadaan badan air (ekstraksi Sentinel-2) terhadap topografi DEMNAS 8m untuk mengevaluasi dampak hidrologi dari aktivitas operasional.
        
        **Interpretasi Data:**
        * **Profil Lahan Alami (Lingkaran Hijau):** Menunjukkan area vegetasi normal yang masih mampu menjaga kelembapan tanah alaminya (NDWI -0.2 hingga 0.0) di sepanjang gradien elevasi.
        * **Anomali Kolam Tambang (Belah Ketupat Biru):** Terdeteksi kluster badan air tidak alami yang membentuk garis horizontal di dataran tinggi (elevasi ~230 mdpl). Ini mengonfirmasi keberadaan infrastruktur penampungan buatan seperti Settling Pond atau Tailing Dam.
        * **Krisis Desikasi Lahan (Segitiga Oranye):** Sebaran masif titik tanah kering kerontang hingga nilai ekstrem (NDWI < -0.2) dari elevasi 0-350 mdpl. Pengupasan topsoil laterit membuat lahan kehilangan kemampuan retensi air dan rawan memicu limpasan permukaan (surface runoff).
        """)
        st.info("**Apa Maksudnya?:**\n\nZona kering kritis (oranye) inilah yang dibidik untuk penaburan Soil Ameliorant sistem COCONUT guna memperbaiki porositas tanah agar kembali mampu mengikat air dan mengunci laju erosi.")

    with col_plot3: # Pastikan nama variabel kolomnya sesuai dengan di kodinganmu
        st.image("grafik_ndwi.png", caption="Data Spasial Aktual: Sentinel-2 (10m) & DEMNAS 8m via Google Earth Engine", use_container_width=True)

    # -------------------------------------------------------
    # GRAFIK KEEMPAT (ELEVASI VS NDMI)
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader(" Analisis Korelasi Spasial: Topografi vs Stres Air Vegetasi (NDMI)")
    st.markdown("Mendeteksi tingkat kekeringan kanopi dan defisit kelembapan lahan akibat perubahan iklim mikro tambang.")

    # Membuat dua kolom baru (Row 4)
    col_text4, col_plot4 = st.columns([1, 1.5])

    with col_text4:
        st.markdown("### 🍂 Elevasi vs NDMI (Water Stress)")
        st.markdown("""
        **Grafik ini memetakan tingkat stres air pada tajuk vegetasi dari satelit Sentinel-2 terhadap zonasi ketinggian DEMNAS 8m untuk mendeteksi degradasi kesehatan hutan. 
        
        **Interpretasi Data:**
        * **Profil Kanopi Sehat (Lingkaran Hijau & Kuning):** Area hutan primer (NDMI > 0.1) dan zona transisi/semak (NDMI 0.0 - 0.1) masih tersebar alami dari pesisir hingga puncak bukit, menunjukkan sisa ekosistem yang mampu mempertahankan kelembapan daunnya.
        * **Anomali Stres Ekstrem (Silang Merah):** Terjadi dominasi dan penumpukan masif titik stres air ekstrem (NDMI < 0) di seluruh rentang topografi (0 - 350 mdpl). Ini membuktikan hilangnya tajuk pohon secara radikal, membuat lahan terekspos panas matahari langsung tanpa perlindungan kanopi (indikasi bukaan tambang dan jalan operasional).
        """)
        st.info("**Apa Maksudnya?:**\n\nDominasi zona stres air (silang merah) ini menegaskan pentingnya intervensi Soil Ameliorant COCONUT. Perbaikan struktur tanah mutlak diperlukan untuk memulihkan iklim mikro (micro-climate) agar bibit revegetasi reklamasi tidak mati kekeringan di lahan kritis tersebut.")

    with col_plot4: # Sesuaikan dengan nama variabel kolom NDMI kamu
        st.image("grafik_ndmi.png", caption="Data Spasial Aktual: Sentinel-2 (10m) & DEMNAS 8m via Google Earth Engine", use_container_width=True)

    # -------------------------------------------------------
    # GRAFIK KELIMA (ELEVASI VS SBI)
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader(" Analisis Korelasi Spasial: Topografi vs Kecerahan Tanah (SBI)")
    st.markdown("Grafik ini memetakan tingkat paparan tanah telanjang (reflektansi cahaya) dari satelit Sentinel-2 terhadap zonasi ketinggian DEMNAS 8m untuk melacak jejak fisik infrastruktur dan bukaan tambang.")

    # Membuat dua kolom baru (Row 5)
    col_text5, col_plot5 = st.columns([1, 1.5])

    with col_text5:
        st.markdown("### ⛏️ Elevasi vs SBI (Soil Brightness)")
        st.markdown("""
        **Savanna/Soil Brightness Index (SBI)** mengukur tingkat kecerahan reflektansi permukaan lahan yang terbuka murni tanpa tutupan tajuk.
        
        **Interpretasi Data:**
        * **Profil Lahan Alami (Lingkaran Hijau & Kuning):** Ekosistem yang sehat akan menyerap cahaya matahari untuk fotosintesis, menghasilkan nilai SBI negatif atau rendah (< 0.1). Titik-titik ini tersebar merata, mewakili sisa hutan yang belum tersentuh.
        * **Anomali Bukaan Tambang (Kotak Ungu):** Terdeteksi lonjakan ekstrem nilai kecerahan tanah (SBI > 0.1) yang memanjang dari pesisir (0 mdpl) hingga menembus perbukitan (300 mdpl). Pola ini adalah bukti tak terbantahkan dari masifnya pengupasan lahan untuk open-pit mining, pembuatan haul road, serta tumpukan stockpile dan slag yang memantulkan cahaya sangat kuat. Bahkan, terdapat titik anomali sangat ekstrem (SBI mendekati 0.5) di elevasi rendah yang mengindikasikan tapak kawasan industri padat.
        """)
        st.info("**Apa Maksudnya?:**\n\nTanah telanjang (ungu) ini telah kehilangan lapisan topsoil alaminya. Area kritis ini merupakan target utama aplikasi pupuk silika/karbonat (Soil Ameliorant) COCONUT untuk merekonstruksi tanah buatan, menurunkan tingkat reflektansi permukaan, dan mempercepat keberhasilan reklamasi tambang.")

    with col_plot5: # Pastikan nama variabelnya sesuai (col_plot5 atau yang kamu pakai)
        st.image("grafik_sbi.png", caption="Data Spasial Aktual: Sentinel-2 (10m) & DEMNAS 8m via Google Earth Engine", use_container_width=True)

# ==========================================
# TAB 3: PETA FOLIUM & GEOSPATIAL ADVANCED COMPUTATION
# ==========================================
with tab3:
    st.subheader("🌍 Web-GIS & Komputasi Spasial Aktif Kawasi, Pulau Obi")
    st.info("Algoritma: Cloud Masking QA60, Deskriptor Hidrologi DEM, dan Ekstraksi Piksel reduceRegion.")

    # Inisialisasi Peta
    m = folium.Map(location=[-1.56, 127.38], zoom_start=12)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False
    ).add_to(m)

    # Batas Area Studi (Buffer 10 Km dari Titik Sentral Kawasi)
    kawasi_roi = ee.Geometry.Point([127.38, -1.56]).buffer(10000)
    
    # -------------------------------------------------------
    # 1. UPGRADE NO 1: ALGORITMA MASKING AWAN (QA60 BITWISE)
    # -------------------------------------------------------
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(kawasi_roi) \
        .filterDate('2023-01-01', '2024-01-01') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_s2_clouds) \
        .median() \
        .clip(kawasi_roi)
    
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # -------------------------------------------------------
    # 2. HIDROLOGI & TOPOGRAFI DINAMIS (DEMNAS GABUNGAN)
    # -------------------------------------------------------
    # Ambil data DEMNAS resolusi tinggi (8m) dari GEE Assets menggantikan SRTM lama
    dem = ee.Image('projects/obi-project-495300/assets/DEMNAS_Obi').clip(kawasi_roi)
    slope = ee.Terrain.slope(dem)
    safe_slope = slope.lt(15) # Lereng aman di bawah 15 derajat tetap aktif
    hillshade = ee.Terrain.hillshade(dem) # Fitur Baru: Efek bayangan relief 3D

    # Klasifikasi Air Permukaan JRC Global Surface Water
    water_mask = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence').gt(0)
    
    # Transformasi jarak hidro-spasial (Buffer Aman > 200 meter dari air)
    dist_to_water = water_mask.fastDistanceTransform().sqrt().multiply(ee.Image.pixelArea().sqrt())
    safe_water = dist_to_water.gt(200)

    # -------------------------------------------------------
    # 3. INTEGRASI MULTI-KRITERIA (ZONASI LALU HITUNG LUAS)
    # -------------------------------------------------------
    # Menggabungkan Lereng Aman AND Jarak Air Aman
    erw_suitability = safe_slope.And(safe_water)
    erw_final_masked = erw_suitability.updateMask(erw_suitability)

    # Komputasi Statistik Piksel untuk Ekstraksi Luas Lahan Aktif (Dari Colab Kamu!)
    pixel_area = erw_suitability.multiply(ee.Image.pixelArea()).rename('area_aman')
    stats = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=kawasi_roi,
        scale=8,
        maxPixels=1e9
    )
    
    # Ambil data meter persegi dari server Google Engine
    luas_m2 = ee.Number(stats.get('area_aman'))
    luas_m2_aman = ee.Algorithms.If(luas_m2, luas_m2, ee.Number(0))
    
    # Konversi m2 ke Hektar secara real-time
    luas_lahan_kalkulasi_ha = float(ee.Number(luas_m2_aman).divide(10000).getInfo())

    # -------------------------------------------------------
    # 4. SINKRONISASI AKTIF KE SIDEBAR & NOTIFIKASI
    # -------------------------------------------------------
    st.markdown("### Hasil Analisis Algoritma Spasial ( reduceRegion )")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Luas Lahan Aman Tersedia (Hasil Satelit)", value=f"{luas_lahan_kalkulasi_ha:,.2f} Ha")
    with metric_col2:
        st.metric(label="Kebutuhan Lahan untuk Volume Input", value=f"{dosis_aplikasi_ha:,.2f} Ha")
    with metric_col3:
        # Menilai kecukupan spasial secara matematis
        if luas_lahan_kalkulasi_ha >= dosis_aplikasi_ha:
            st.success("✅ Daya Dukung Lahan: CUKUP")
        else:
            st.error("❌ Daya Dukung Lahan: KURANG")

    # -------------------------------------------------------
    # 5. RENDERING MAP LAYERS
    # -------------------------------------------------------
    ndvi_vis = {'min': 0.0, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
    add_ee_layer(m, hillshade, {'min': 0, 'max': 255}, 'Relief Topografi 3D (DEMNAS)', opacity=0.5)
    add_ee_layer(m, ndvi, ndvi_vis, 'Indeks Vegetasi Bersih (Cloud-Free NDVI)')
    add_ee_layer(m, water_mask.updateMask(water_mask), {'palette': ['blue']}, 'Sumber Air Permukaan')
    add_ee_layer(m, slope.updateMask(slope.gte(15)), {'min': 15, 'max': 60, 'palette': ['red']}, 'Slope Berbahaya (>15°)')
    add_ee_layer(m, erw_final_masked, {'palette': ['00FF00']}, 'Zona Rekomendasi ERW')

    folium.LayerControl().add_to(m)
    st_folium(m, width=800, height=550)