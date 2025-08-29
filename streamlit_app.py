import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Süper Lig Dashboard", page_icon="⚽", layout="wide")

STATS_URL = "https://fbref.com/en/comps/36/Super-Lig-Stats"

@st.cache_data(show_spinner=False)
def load_data_from_fbref(url: str) -> pd.DataFrame:
    """FBref'ten lig tablosunu okumayı dener, başarısız olursa None döner."""
    try:
        tables = pd.read_html(url)
        df = tables[0].copy()
        # İhtiyaç duyacağımız kolonlar
        keep = ['Rk','Squad','MP','W','D','L','GF','GA','GD','Pts']
        df = df[keep]
        # Numerik kolonları dönüştür
        for c in ['Rk','MP','W','D','L','GF','GA','GD','Pts']:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Temizlik
        df = df.dropna(subset=['Squad']).reset_index(drop=True)
        return df
    except Exception:
        return None

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ['Rk','MP','W','D','L','GF','GA','GD','Pts']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def radar_chart(ax, values, labels, title=""):
    # Radar için kapanış
    values = list(values)
    values += values[:1]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title)

# ---------- UI ----------
st.title("⚽ Süper Lig Takım Dashboard")

st.caption("Kaynak: FBref (çekilemezse CSV yükleyebilirsiniz).")

# Veri getirme
with st.spinner("Veriler yükleniyor..."):
    team_stats = load_data_from_fbref(STATS_URL)

# Fallback: CSV yükleme
if team_stats is None:
    st.warning("FBref'e bağlanılamadı veya tablo okunamadı. Aşağıdan CSV yükleyin.")
    uploaded = st.file_uploader("Takım istatistikleri CSV yükle (Rk,Squad,MP,W,D,L,GF,GA,GD,Pts)", type=["csv"])
    if uploaded is not None:
        team_stats = pd.read_csv(uploaded)
        team_stats = ensure_numeric(team_stats)

# Örnek CSV ver (test için)
if team_stats is None:
    st.info("Hızlı test için örnek veri kullanılıyor.")
    sample = {
        "Rk":[1,2,3],
        "Squad":["Galatasaray","Trabzonspor","Göztepe"],
        "MP":[3,3,3],
        "W":[3,3,2],
        "D":[0,0,1],
        "L":[0,0,0],
        "GF":[10,3,5],
        "GA":[0,0,0],
        "GD":[10,3,5],
        "Pts":[9,9,7]
    }
    team_stats = pd.DataFrame(sample)

# Veri hazır mı?
if team_stats is not None and len(team_stats) > 0:
    # Sol panel: takım seçimi ve sıralama
    left, right = st.columns([1,2])

    with left:
        st.subheader("Takım Seçimi")
        team = st.selectbox("Takım", team_stats['Squad'].tolist())

        st.markdown("**Sıralama Tablosu**")
        sort_by = st.selectbox("Sırala", ["Pts","GD","GF","GA","W","L","MP"])
        asc = st.toggle("Artan sırala mı?", value=False)
        st.dataframe(team_stats.sort_values(by=sort_by, ascending=asc), use_container_width=True)

    with right:
        st.subheader("Takım Özeti")
        team_row = team_stats[team_stats["Squad"] == team].iloc[0]

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Puan", int(team_row["Pts"]))
        kpi2.metric("Gol Farkı", int(team_row["GD"]))
        kpi3.metric("Atılan Gol", int(team_row["GF"]))
        kpi4.metric("Yenilen Gol", int(team_row["GA"]))
        kpi5.metric("Maç", int(team_row["MP"]))

        st.divider()
        st.subheader("Görselleştirmeler")

        # Bar chart: Pts & GD
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.bar(["Puan","Gol Farkı"], [team_row["Pts"], team_row["GD"]])
        ax1.set_ylim(0, max(team_stats["Pts"].max(), team_stats["GD"].max()) + 5)
        ax1.set_title(f"{team} • Puan & Gol Farkı")
        st.pyplot(fig1)

        # Radar: W,D,L,GF,GA
        radar_labels = ["W","D","L","GF","GA"]
        radar_vals = [float(team_row[l]) for l in radar_labels]
        fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        radar_chart(ax2, radar_vals, radar_labels, title=f"{team} • Radar")
        st.pyplot(fig2)

        st.divider()

        # Lig geneli: Puan grafiği
        st.subheader("Lig Genel Puan Dağılımı")
        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.bar(team_stats["Squad"], team_stats["Pts"])
        ax3.set_xticklabels(team_stats["Squad"], rotation=45, ha="right")
        ax3.set_ylabel("Puan")
        ax3.set_title("Lig • Puanlar")
        st.pyplot(fig3)

else:
    st.error("Veri yüklenemedi. CSV yüklemeyi deneyin.")
