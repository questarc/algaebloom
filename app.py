import streamlit as st
import pandas as pd
import plotly.express as px
import random

# -----------------------------
# Helper functions
# -----------------------------

TOXIN_MOTIFS = [
    "ATGACCTGACCTG",
    "CGTACGTACGTA",
    "GGCATGGCATGG"
]

def classify_sequence(seq: str):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    if not seq:
        return "Unknown", []

    hits = [m for m in TOXIN_MOTIFS if m in seq]
    if hits:
        return "Harmful", hits
    else:
        return "Neutral", []

def design_crispr_guide(seq: str, length: int = 20):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    cleaned = "".join([b for b in seq if b in "ACGT"])
    if len(cleaned) < length:
        return None
    return cleaned[:length]

def reverse_complement(seq: str):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement.get(b, "N") for b in seq[::-1])

def design_primers(seq: str, primer_len: int = 18):
    seq = seq.upper().replace(" ", "").replace("\n", "")
    cleaned = "".join([b for b in seq if b in "ACGT"])
    if len(cleaned) < 2 * primer_len + 20:
        return None, None
    forward = cleaned[:primer_len]
    reverse = reverse_complement(cleaned[-primer_len:])
    return forward, reverse

def simulate_sherlock_signal(guide: str, target: str):
    if not guide or not target:
        return 0.0

    guide = guide.upper().replace(" ", "").replace("\n", "")
    target = target.upper().replace(" ", "").replace("\n", "")

    if len(target) < len(guide):
        return 0.0

    window = target[:len(guide)]
    matches = sum(1 for g, t in zip(guide, window) if g == t)
    score = matches / len(guide)
    return round(score * 100, 1)

def generate_example_dataset(n=20):
    data = []
    for i in range(n):
        base_seq = "".join(random.choice("ACGT") for _ in range(200))
        is_harmful = random.random() < 0.4

        if is_harmful:
            motif = random.choice(TOXIN_MOTIFS)
            pos = random.randint(0, len(base_seq) - len(motif))
            seq = base_seq[:pos] + motif + base_seq[pos:]
        else:
            seq = base_seq

        label, _ = classify_sequence(seq)
        data.append({
            "Sample ID": f"S{i+1}",
            "Sequence": seq,
            "Label": label
        })

    return pd.DataFrame(data)

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(
    page_title="Algaebloom – Synthetic Biology CRISPR App",
    layout="wide"
)

st.title("Algaebloom: Synthetic Biology CRISPR-SHERLOCK Explorer")

st.markdown("""
Algaebloom is a conceptual synthetic biology dashboard for **algal bloom monitoring**.

It lets you:

- Upload or paste DNA sequences  
- Classify them as **harmful** or **neutral**  
- Design a **CRISPR-SHERLOCK assay**  
- Visualize bloom risk and assay signals  
""")

st.sidebar.header("About Algaebloom")
st.sidebar.markdown("""
**Goal:** Explore how synthetic biology and CRISPR diagnostics can detect harmful algal blooms.

This app uses **synthetic motifs and simplified models** for educational purposes.
""")

tab_overview, tab_classification, tab_crispr, tab_visuals = st.tabs(
    ["Overview", "Sequence classification", "CRISPR-SHERLOCK design", "Visualizations"]
)

# -----------------------------
# Overview Tab
# -----------------------------
with tab_overview:
    st.subheader("Biology Background")
    st.markdown("""
Harmful algal blooms (HABs) produce toxins that damage ecosystems and human health.  
Neutral blooms do not produce toxins and are part of natural cycles.
""")

    st.subheader("CRISPR-SHERLOCK Concept")
    st.markdown("""
SHERLOCK uses CRISPR enzymes to detect specific DNA/RNA sequences with high sensitivity.

This app simulates:

1. Guide RNA design  
2. Primer design  
3. Assay signal simulation  
""")

# -----------------------------
# Sequence Classification Tab
# -----------------------------
with tab_classification:
    st.subheader("Classify DNA Sequences")

    default_seq = """>Sample_1
ATGACCTGACCTGTTTACGATCGTAGCTAGCTAGCTAGCTAGCTGACTGACTGACTG
>Sample_2
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA
"""

    seq_text = st.text_area("Input sequences", value=default_seq, height=200)

    if st.button("Classify sequences"):
        sequences = []
        header = None
        seq_buffer = []

        for line in seq_text.strip().splitlines():
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences.append((header, "".join(seq_buffer)))
                header = line[1:]
                seq_buffer = []
            else:
                seq_buffer.append(line)

        if header:
            sequences.append((header, "".join(seq_buffer)))

        results = []
        for header, seq in sequences:
            label, motifs = classify_sequence(seq)
            results.append({
                "Sample ID": header,
                "Length": len(seq),
                "Classification": label,
                "Motifs found": ", ".join(motifs) if motifs else "None"
            })

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        counts = df_results["Classification"].value_counts().reset_index()
        counts.columns = ["Classification", "Count"]

        fig = px.bar(counts, x="Classification", y="Count", color="Classification",
                     title="Harmful vs Neutral Sequences", text="Count")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CRISPR-SHERLOCK Tab
# -----------------------------
with tab_crispr:
    st.subheader("CRISPR Assay Designer")

    target_default = "ATGACCTGACCTGTTTACGATCGTAGCTAGCTAGCTAGCTAGCTGACTGACTGACTG"
    target_seq = st.text_area("Target DNA Sequence", value=target_default, height=150)

    col1, col2 = st.columns(2)
    with col1:
        guide_length = st.slider("Guide length", 15, 24, 20)
    with col2:
        primer_length = st.slider("Primer length", 16, 24, 18)

    if st.button("Design CRISPR assay"):
        guide = design_crispr_guide(target_seq, guide_length)
        fwd, rev = design_primers(target_seq, primer_length)

        if guide:
            st.markdown("### Guide RNA")
            st.code(guide)
        else:
            st.error("Guide could not be designed.")

        if fwd and rev:
            st.markdown("### Primers")
            st.code(f"Forward: {fwd}")
            st.code(f"Reverse: {rev}")
        else:
            st.warning("Primers could not be designed.")

        if guide:
            signal = simulate_sherlock_signal(guide, target_seq)
            st.markdown(f"### Simulated SHERLOCK Signal: **{signal} a.u.**")

            fig_signal = px.bar(x=["Signal"], y=[signal], range_y=[0, 100],
                                title="Simulated CRISPR-SHERLOCK Signal", text=[signal])
            st.plotly_chart(fig_signal, use_container_width=True)

# -----------------------------
# Visualizations Tab
# -----------------------------
with tab_visuals:
    st.subheader("Synthetic Dataset Visualization")

    n_samples = st.slider("Number of samples", 10, 200, 40)

    if st.button("Generate synthetic dataset"):
        df_samples = generate_example_dataset(n_samples)
        st.dataframe(df_samples, use_container_width=True)

        label_counts = df_samples["Label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]

        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(label_counts, names="Label", values="Count",
                             title="Harmful vs Neutral Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            signals = []
            for _, row in df_samples.iterrows():
                if row["Label"] == "Harmful":
                    signals.append(random.uniform(60, 100))
                else:
                    signals.append(random.uniform(0, 40))

            df_samples["Signal"] = signals

            fig_hist = px.histogram(df_samples, x="Signal", color="Label",
                                    nbins=20, title="Simulated Assay Signal Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f0f2f6;
            color: #333;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            border-top: 1px solid #d3d3d3;
        }
    </style>
    <div class="footer">
        App Developed by <strong>Sathvik Kakarla</strong>
    </div>
    """,
    unsafe_allow_html=True
)
