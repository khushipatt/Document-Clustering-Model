"""
app.py — Document Clustering Streamlit App
Algorithms : K-Means | Hierarchical (Agglomerative) | LDA Topic Modeling
Visuals    : Word Clouds | Cluster Bar Chart | Elbow Method Chart
Upload     : CSV or plain TXT files
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

from clustering import (
    load_from_csv, load_from_txt,
    vectorize,
    kmeans_cluster, hierarchical_cluster, lda_cluster,
    get_top_words, get_top_words_from_labels,
    compute_silhouette, elbow_method,
    reduce_to_2d
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Clustering",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d3748;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 2rem 0 1rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📄 Document Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your documents and discover hidden topic groups using ML clustering algorithms.</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # File Upload
    st.markdown("### 📁 Upload Data")
    upload_type = st.radio("File Type", ["CSV", "TXT"], horizontal=True)

    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["csv", "txt"],
        help="CSV: must have a text column. TXT: one document per paragraph."
    )

    text_col = None
    if upload_type == "CSV" and uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            text_col = st.selectbox(
                "Select Text Column",
                options=preview_df.columns.tolist(),
                help="Choose the column containing document text"
            )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.markdown("---")

    # Algorithm
    st.markdown("### 🧠 Algorithm")
    algorithm = st.selectbox(
        "Clustering Method",
        ["K-Means", "Hierarchical (Agglomerative)", "LDA (Topic Modeling)"],
        help="K-Means: fast & popular. Hierarchical: tree-based. LDA: probabilistic topic model."
    )

    if algorithm == "Hierarchical (Agglomerative)":
        linkage_type = st.selectbox(
            "Linkage Type", ["ward", "average", "complete", "single"],
            help="Ward minimizes variance; average uses mean distances."
        )
    else:
        linkage_type = "ward"

    n_clusters = st.slider("Number of Clusters / Topics", 2, 12, 4)

    st.markdown("---")

    # Advanced
    st.markdown("### 🔧 Advanced")
    max_features = st.slider("Max TF-IDF Features", 500, 5000, 2000, 500)
    top_n_words = st.slider("Top Words per Cluster", 5, 20, 10)
    max_docs = st.slider("Max Documents to Process", 50, 5000, 500, 50,
                         help="Limit to speed up processing")

    st.markdown("---")
    run_btn = st.button("🚀 Run Clustering", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>Built for college assignment 🎓<br>"
        "Algorithms: K-Means · Hierarchical · LDA</small>",
        unsafe_allow_html=True
    )

# ─── Sample Data Helper ───────────────────────────────────────────────────────
SAMPLE_DOCS = [
    # Tech
    "Machine learning algorithms enable computers to learn from data without explicit programming.",
    "Deep learning neural networks have revolutionized computer vision and natural language processing.",
    "Python is the most popular programming language for data science and artificial intelligence.",
    "Cloud computing platforms like AWS and Azure provide scalable infrastructure for applications.",
    "Cybersecurity threats are growing as organizations increasingly rely on digital infrastructure.",
    "Blockchain technology enables decentralized and transparent record-keeping across networks.",
    "The Internet of Things connects billions of devices enabling smart home and industrial automation.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
    "Software developers use agile methodologies to deliver software in iterative cycles.",
    "Open source software has transformed how developers collaborate and build applications.",
    # Sports
    "The football team won the championship after a thrilling overtime match in the stadium.",
    "Basketball players train intensively to improve their shooting accuracy and defensive skills.",
    "The tennis tournament attracted top players from around the world for the grand slam event.",
    "Soccer is the most popular sport globally with billions of fans across every continent.",
    "Olympic athletes dedicate years of training to compete for gold medals in their events.",
    "Cricket is a beloved sport in South Asia with passionate fans supporting national teams.",
    "The marathon runner crossed the finish line after pushing through 26 miles of grueling effort.",
    "Swimming coaches focus on technique and endurance to help athletes break world records.",
    "Baseball statistics have become increasingly sophisticated with advanced sabermetrics analysis.",
    "Rugby matches require players to combine strength, speed, and tactical teamwork effectively.",
    # Health
    "Regular exercise and a balanced diet are key components of maintaining good physical health.",
    "Mental health awareness campaigns have helped reduce stigma around seeking therapy and counseling.",
    "Vaccines have been instrumental in eradicating diseases like smallpox and reducing polio cases.",
    "The healthcare system faces challenges from aging populations and rising treatment costs globally.",
    "Yoga and meditation practices have been shown to reduce stress and improve mental wellbeing.",
    "Research into Alzheimer's disease is advancing with new drug trials showing promising results.",
    "Nutrition science reveals how gut microbiome diversity impacts immunity and overall health.",
    "Doctors recommend regular screenings to detect cancer early when treatment is most effective.",
    "Sleep deprivation negatively affects cognitive function memory and immune system performance.",
    "Telemedicine enables patients to consult doctors remotely improving access to healthcare.",
    # Finance
    "Stock market volatility increased sharply following the central bank interest rate announcement.",
    "Cryptocurrency markets are highly speculative with Bitcoin and Ethereum leading trading volumes.",
    "Personal finance advisors recommend building an emergency fund equivalent to six months expenses.",
    "Inflation erodes purchasing power making it essential to invest savings rather than hold cash.",
    "Venture capital firms invest millions in early stage startups with high growth potential.",
    "Real estate remains a popular investment due to its tangibility and potential rental income.",
    "ETFs provide diversified exposure to markets at lower costs compared to actively managed funds.",
    "Compound interest is the most powerful concept in personal finance for long-term wealth building.",
    "Banks are adopting artificial intelligence to detect fraudulent transactions in real time.",
    "Economic recessions are characterized by declining GDP employment and consumer spending.",
    # Science
    "NASA scientists discovered evidence of ancient water activity on the surface of Mars.",
    "Climate change is causing sea levels to rise threatening coastal communities worldwide.",
    "CRISPR gene editing technology allows scientists to precisely modify DNA sequences in organisms.",
    "Astronomers detected a new exoplanet in the habitable zone of a nearby star system.",
    "Renewable energy sources like solar and wind power are rapidly replacing fossil fuels globally.",
    "Particle physicists at CERN are studying the fundamental building blocks of the universe.",
    "Ocean acidification caused by carbon dioxide absorption is threatening coral reef ecosystems.",
    "Neuroscientists are mapping brain connectivity to understand consciousness and memory formation.",
    "Stem cell research offers hope for regenerating damaged tissues and treating degenerative diseases.",
    "The James Webb Space Telescope has revealed unprecedented details about distant galaxies.",
]


def get_sample_data():
    return SAMPLE_DOCS.copy()


# ─── Main App ─────────────────────────────────────────────────────────────────

# Show upload instructions when no file
if not run_btn and uploaded_file is None:
    st.markdown('<div class="section-header">📋 How to Use</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1 — Upload**
        - Upload a **CSV** file with a text column, or
        - Upload a **TXT** file with paragraphs as separate documents
        - Or click Run to try with **built-in sample data**
        """)
    with col2:
        st.markdown("""
        **Step 2 — Configure**
        - Pick your clustering algorithm
        - Choose number of clusters
        - Adjust advanced settings
        """)
    with col3:
        st.markdown("""
        **Step 3 — Explore Results**
        - View cluster scatter plot
        - Explore word clouds
        - Analyze bar charts & elbow curve
        - Download cluster assignments
        """)

    st.markdown('<div class="section-header">📊 Sample CSV Format</div>', unsafe_allow_html=True)
    sample_preview = pd.DataFrame({
        "id": [1, 2, 3],
        "text": [
            "Machine learning is transforming industries...",
            "The football team won the championship...",
            "Vaccines have helped eradicate many diseases..."
        ],
        "category": ["tech", "sports", "health"]
    })
    st.dataframe(sample_preview, use_container_width=True)
    st.info("💡 No file? Click **Run Clustering** with no upload to try on built-in sample data (50 documents, 5 topics).")

# ─── Run Pipeline ─────────────────────────────────────────────────────────────
if run_btn:

    # ── Load Data ──────────────────────────────────────────────
    docs = []
    source_label = ""

    if uploaded_file is not None:
        try:
            if upload_type == "CSV":
                df_raw = pd.read_csv(uploaded_file)
                if text_col is None or text_col not in df_raw.columns:
                    st.error("❌ Please select a valid text column from the sidebar.")
                    st.stop()
                docs = load_from_csv(df_raw, text_col)
                source_label = f"CSV → `{text_col}` column"
            else:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
                docs = load_from_txt(content)
                source_label = "TXT file"
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    else:
        docs = get_sample_data()
        source_label = "Built-in sample data (5 topics × 10 docs)"

    # Limit docs
    docs = [d for d in docs if len(d.strip()) > 15][:max_docs]

    if len(docs) < max(n_clusters * 2, 10):
        st.error(f"❌ Not enough documents ({len(docs)}). Need at least {max(n_clusters*2, 10)} for {n_clusters} clusters. Try fewer clusters or upload more data.")
        st.stop()

    # ── Vectorize ──────────────────────────────────────────────
    with st.spinner("🔄 Vectorizing documents with TF-IDF..."):
        X, vectorizer, cleaned_docs = vectorize(docs, max_features=max_features)

    # ── Cluster ────────────────────────────────────────────────
    with st.spinner(f"🔄 Running {algorithm} clustering..."):
        if algorithm == "K-Means":
            cluster_labels, model = kmeans_cluster(X, n_clusters)
            top_words = get_top_words(model, vectorizer, top_n_words, method='kmeans')
            method_key = 'kmeans'

        elif algorithm == "Hierarchical (Agglomerative)":
            cluster_labels, model = hierarchical_cluster(X, n_clusters, linkage_type)
            top_words = get_top_words_from_labels(X, cluster_labels, vectorizer, top_n_words)
            method_key = 'hierarchical'

        else:  # LDA
            cluster_labels, model = lda_cluster(X, n_clusters)
            top_words = get_top_words(model, vectorizer, top_n_words, method='lda')
            method_key = 'lda'

    # ── Silhouette ─────────────────────────────────────────────
    with st.spinner("📏 Computing silhouette score..."):
        sil_score = compute_silhouette(X, cluster_labels)

    # ── 2D Reduction ───────────────────────────────────────────
    with st.spinner("📐 Reducing dimensions for visualization..."):
        coords = reduce_to_2d(X)

    # ─────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────
    st.success(f"✅ Clustering complete! Source: {source_label}")
    st.markdown("---")

    # ── Metrics Row ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Documents", len(docs))
    m2.metric("🔗 Clusters", n_clusters)
    m3.metric("📐 Silhouette Score", f"{sil_score:.4f}",
              help="Range: -1 to 1. Higher = better defined clusters.")
    m4.metric("🔤 Vocabulary Size", f"{X.shape[1]:,}")

    # ── Build DataFrame ────────────────────────────────────────
    result_df = pd.DataFrame({
        "document_id": range(len(docs)),
        "cluster": cluster_labels,
        "preview": [d[:120] + "..." if len(d) > 120 else d for d in cleaned_docs],
        "original": [d[:200] + "..." if len(d) > 200 else d for d in docs]
    })

    # ── Color Palette ──────────────────────────────────────────
    COLOR_SEQ = px.colors.qualitative.Bold
    cluster_colors = {i: COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(n_clusters)}

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: Scatter Plot
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🗺️ Cluster Scatter Plot (PCA 2D)</div>', unsafe_allow_html=True)

    df_scatter = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'Cluster': [f"Cluster {c}" for c in cluster_labels],
        'Preview': result_df['preview']
    })
    fig_scatter = px.scatter(
        df_scatter, x='PC1', y='PC2',
        color='Cluster',
        hover_data={'Preview': True, 'PC1': ':.2f', 'PC2': ':.2f'},
        color_discrete_sequence=COLOR_SEQ,
        title=f"Document Clusters — {algorithm} (n={n_clusters})",
        template="plotly_white"
    )
    fig_scatter.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color='white')))
    fig_scatter.update_layout(
        legend=dict(orientation='v', x=1.02, y=0.5),
        margin=dict(r=160),
        height=480
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: Cluster Bar Chart
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📊 Cluster Distribution</div>', unsafe_allow_html=True)

    bar_col1, bar_col2 = st.columns([2, 1])

    with bar_col1:
        dist_counts = pd.Series(cluster_labels).value_counts().sort_index()
        dist_df = pd.DataFrame({
            'Cluster': [f"Cluster {i}" for i in dist_counts.index],
            'Documents': dist_counts.values,
            'Percentage': [f"{v/len(docs)*100:.1f}%" for v in dist_counts.values]
        })

        fig_bar = go.Figure(go.Bar(
            x=dist_df['Cluster'],
            y=dist_df['Documents'],
            text=dist_df['Percentage'],
            textposition='outside',
            marker=dict(
                color=[COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(dist_df))],
                line=dict(width=1.5, color='white'),
                cornerradius=6
            )
        ))
        fig_bar.update_layout(
            title="Number of Documents per Cluster",
            xaxis_title="Cluster",
            yaxis_title="Document Count",
            template="plotly_white",
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with bar_col2:
        # Pie chart
        fig_pie = px.pie(
            dist_df, values='Documents', names='Cluster',
            color_discrete_sequence=COLOR_SEQ,
            hole=0.4,
            title="Cluster Share"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            showlegend=False,
            height=380,
            template="plotly_white",
            margin=dict(t=50, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: Word Clouds
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">☁️ Word Clouds per Cluster</div>', unsafe_allow_html=True)

    CMAPS = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges',
             'YlOrBr', 'PuBuGn', 'RdPu', 'BuPu', 'GnBu']

    n_cols = min(n_clusters, 4)
    wc_rows = (n_clusters + n_cols - 1) // n_cols

    for row in range(wc_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            cluster_id = row * n_cols + col_idx
            if cluster_id >= n_clusters:
                break
            words = top_words.get(cluster_id, [])
            if not words:
                continue
            word_text = " ".join(words * 3)  # repeat for density
            with cols[col_idx]:
                try:
                    wc = WordCloud(
                        width=400, height=260,
                        background_color='white',
                        colormap=CMAPS[cluster_id % len(CMAPS)],
                        prefer_horizontal=0.8,
                        max_words=50,
                        collocations=False
                    ).generate(word_text)

                    fig_wc, ax = plt.subplots(figsize=(4, 2.6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    fig_wc.patch.set_facecolor('white')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)

                    # Top words as tags
                    tags = " · ".join(words[:6])
                    st.markdown(
                        f"<div style='text-align:center; font-weight:700; "
                        f"color:{COLOR_SEQ[cluster_id % len(COLOR_SEQ)]}; font-size:0.9rem;'>"
                        f"Cluster {cluster_id}</div>"
                        f"<div style='text-align:center; color:#555; font-size:0.75rem;'>{tags}</div>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.warning(f"Could not generate word cloud for Cluster {cluster_id}: {e}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: Elbow Method (K-Means only)
    # ═══════════════════════════════════════════════════════════
    if algorithm == "K-Means":
        st.markdown('<div class="section-header">📉 Elbow Method — Find Optimal K</div>', unsafe_allow_html=True)

        with st.spinner("Computing elbow curve (K=2 to 10)..."):
            ks, inertias, silhouettes = elbow_method(X, max_k=min(10, len(docs) // 2))

        el1, el2 = st.columns(2)

        with el1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=ks, y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#667eea', width=3),
                marker=dict(size=9, symbol='circle', color='#667eea',
                            line=dict(width=2, color='white'))
            ))
            fig_elbow.add_vline(
                x=n_clusters, line_dash="dash", line_color="#e53e3e",
                annotation_text=f"K={n_clusters} (chosen)",
                annotation_position="top right"
            )
            fig_elbow.update_layout(
                title="Elbow Method — Inertia vs K",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Inertia (Within-cluster sum of squares)",
                template="plotly_white",
                height=360
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

        with el2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=ks, y=silhouettes,
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='#48bb78', width=3),
                marker=dict(size=9, symbol='diamond', color='#48bb78',
                            line=dict(width=2, color='white'))
            ))
            fig_sil.add_vline(
                x=n_clusters, line_dash="dash", line_color="#e53e3e",
                annotation_text=f"K={n_clusters} (chosen)",
                annotation_position="top right"
            )
            fig_sil.update_layout(
                title="Silhouette Score vs K (Higher = Better)",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Silhouette Score",
                template="plotly_white",
                height=360
            )
            st.plotly_chart(fig_sil, use_container_width=True)

        best_k = ks[silhouettes.index(max(silhouettes))]
        st.info(f"💡 **Suggested K = {best_k}** based on highest silhouette score ({max(silhouettes):.4f}). "
                f"You chose K = {n_clusters}.")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: Top Keywords Table
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🔑 Top Keywords per Cluster</div>', unsafe_allow_html=True)

    kw_data = {}
    for cid, words in top_words.items():
        kw_data[f"Cluster {cid}"] = words[:top_n_words]
    kw_df = pd.DataFrame(kw_data).T
    kw_df.columns = [f"#{i+1}" for i in range(kw_df.shape[1])]
    st.dataframe(kw_df, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # SECTION 6: Sample Documents Explorer
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📝 Explore Documents by Cluster</div>', unsafe_allow_html=True)

    sel_cluster = st.selectbox(
        "Select Cluster",
        options=list(range(n_clusters)),
        format_func=lambda x: f"Cluster {x}  (top words: {', '.join(top_words.get(x, [])[:4])})"
    )

    cluster_docs = result_df[result_df['cluster'] == sel_cluster]
    st.markdown(f"**{len(cluster_docs)} documents** in Cluster {sel_cluster}")

    for _, row in cluster_docs.head(5).iterrows():
        with st.expander(f"Doc #{row['document_id']}  — {row['preview'][:60]}..."):
            st.write(row['original'])

    # ═══════════════════════════════════════════════════════════
    # SECTION 7: Download Results
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">⬇️ Download Results</div>', unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)

    with dl1:
        # Full results CSV
        export_df = pd.DataFrame({
            "document_id": range(len(docs)),
            "cluster": cluster_labels,
            "text_preview": [d[:200] for d in docs]
        })
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Cluster Assignments (CSV)",
            data=csv_bytes,
            file_name="cluster_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with dl2:
        # Top words CSV
        kw_export = kw_df.reset_index().rename(columns={'index': 'cluster'})
        kw_csv = kw_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Top Keywords (CSV)",
            data=kw_csv,
            file_name="cluster_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
