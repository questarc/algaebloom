import streamlit as st
import pandas as pd
import plotly.express as px
import random
from textwrap import wrap

# -----------------------------
# Helper functions
# -----------------------------

TOXIN_MOTIFS = [
    "ATGACCTGACCTG",  # synthetic "toxin" motif 1
    "CGTACGTACGTA",   # synthetic "toxin" motif 2
    "GGCATGGCATGG"    # synthetic "toxin" motif 3
]

def classify_sequence(seq: str):
    """
    Very simplified classification:
    If any TOXIN_MOTIF appears in the sequence, call it harmful.
    """
    seq = seq.upper().replace(" ", "").replace("\n", "")
    if not seq:
        return "Unknown", []

    hits = [m for m in TOXIN_MOTIFS if m in seq]
    if hits:
        return "Harmful", hits
    else:
        return "Neutral", []

def design_crispr_guide(seq: str, length: int = 20):
    """
    Simplified guide design: take the first 'length' bases
    that look valid (A/C/G/T) and return it.
    """
    seq = seq.upper().replace(" ", "").replace("\n", "")
    cleaned = "".join([b for b in seq if b in "ACGT"])
    if len(cleaned) < length:
        return None
    return cleaned[:length]

def design_primers(seq: str, primer_len: int = 18):
    """
    Very naive primer design: pick 5' and 3' segments of the sequence.
    """
    seq = seq.upper().replace(" ", "").replace("\n", "")
    cleaned = "".join([b for b in seq if b in "ACGT"])
    if len(cleaned) < 2 * primer_len + 20:
        return None, None
    forward = cleaned[:primer_len]
    reverse = reverse_complement(cleaned[-primer_len:])
    return forward, reverse

def reverse_complement(seq: str):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement.get(b, "N") for b in seq[::-1])

def simulate_sherlock_signal(guide: str, target: str):
    """
    Toy model of SHERLOCK signal:
    - Score based on number of matching bases between guide and target.
    - Normalize to [0, 1] and scale to 0–100.
    """
    if not guide or not target:
        return 0.0

    guide = guide.upper().replace(" ", "").replace("\n", "")
    target = target.upper().replace(" ", "").replace("\n", "")

    if len(target) < len(guide):
        return 0.0

    window = target[:len(guide)]
    matches = sum(1 for g, t in zip(guide, window) if g == t)
    score = matches / len(guide)  # fraction of matching bases
    return round(score * 100, 1)

def generate_example_dataset(n=20):
    """
    Create a synthetic dataset of samples with "harmful" or "neutral" labels.
    """
    data = []
    for i in range(n):
        base_seq = "".join(random.choice("ACGT") for _ in range(200))
        is_harmful = random.random() < 0.4  # ~40% harmful

        motifs_found = []
        if is_harmful:
            motif = random.choice(TOXIN_MOTIFS)
            insert_pos = random.randint(0, len(base_seq) - len(motif))
            seq = base_seq[:insert_pos] + motif + base_seq[insert_pos:]
            motifs_found.append(motif)
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
# Streamlit app
# -----------------------------

st.set_page_config(
    page_title="Algaebloom – Synthetic Biology CRISPR App",
    layout="wide"
)

st.title("Algaebloom: Synthetic Biology CRISPR-SHERLOCK Explorer")

st.markdown(
    """
Algaebloom is a conceptual synthetic biology dashboard for **algal bloom monitoring**.
It lets you:

- Upload or paste DNA sequences from algae samples
- Classify them as **harmful** or **neutral** based on toy toxin motifs
- Design a simple **CRISPR-SHERLOCK assay** (guide RNA + primers)
- Visualize sample distributions and simulated assay signals
"""
)

st.sidebar.header("About Algaebloom")
st.sidebar.markdown(
    """
**Goal:** Explore how synthetic biology and CRISPR diagnostics could help
distinguish harmful algal blooms from neutral ones.

**Key ideas:**

- Harmful blooms carry **toxin genes**
- CRISPR-SHERLOCK can detect specific **DNA/RNA sequences**
- Visual analytics help understand bloom risk and assay performance

This app uses **synthetic motifs and toy models**; it is for education only.
**App developed by Sathvik Kakarla**
"""
)

tab_overview, tab_classification, tab_crispr, tab_visuals = st.tabs(
    ["Overview", "Sequence classification", "CRISPR-SHERLOCK design", "Visualizations"]
)

# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    st.subheader("Biology background")

    st.markdown(
        """
**Algal blooms** occur when algae or cyanobacteria grow rapidly in water.
They can be:

- **Harmful algal blooms (HABs):** Produce toxins and damage ecosystems, 
  aquaculture, and human health.
- **Neutral blooms:** Do not produce major toxins; part of normal ecosystem dynamics.
"""
    )

    st.markdown(
        """
In real systems, harmful species may produce:

- Neurotoxins (e.g., domoic acid, saxitoxin)
- Hepatotoxins (e.g., microcystins)
- Other metabolites that affect fish, shellfish, mammals, and humans

DNA- and RNA-based diagnostics can identify **which species and toxin genes** are present,
even when blooms look similar by eye or satellite.
"""
    )

    st.subheader("CRISPR-SHERLOCK concept (simplified)")

    st.markdown(
        """
**SHERLOCK** (Specific High-sensitivity Enzymatic Reporter unLOCKing) uses CRISPR
enzymes with guide RNAs to detect defined nucleic acid targets.

In this app, we:

1. Choose a target sequence (e.g., toxin gene fragment).
2. Design a 20-nt guide RNA sequence.
3. Design simple primers flanking the region.
4. Simulate an **assay signal** based on how well the guide matches the target.
"""
    )

# -----------------------------
# Sequence classification tab
# -----------------------------
with tab_classification:
    st.subheader("Classify sequences as harmful or neutral")

    st.markdown(
        """
Paste one or more DNA sequences (FASTA-like or raw). Each sequence block
can be separated by a line starting with `>` (like FASTA headers).
"""
    )

    default_seq = """>Sample_1
ATGACCTGACCTGTTTACGATCGTAGCTAGCTAGCTAGCTAGCTGACTGACTGACTG
>Sample_2
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA
"""
    seq_text = st.text_area(
        "Input sequences",
        value=default_seq,
        height=200
    )

    if st.button("Classify sequences"):
        sequences = []
        current_header = None
        current_seq = []

        for line in seq_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:] or f"Sample_{len(sequences) + 1}"
                current_seq = []
            else:
                current_seq.append(line)

        # Last sequence
        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))

        results = []
        for header, seq in sequences:
            label, motifs = classify_sequence(seq)
            results.append({
                "Sample ID": header,
                "Length": len(seq.replace(" ", "").replace("\n", "")),
                "Classification": label,
                "Motifs found": ", ".join(motifs) if motifs else "None"
            })

        if results:
            df_results = pd.DataFrame(results)
            st.success(f"Classified {len(results)} sequence(s).")
            st.dataframe(df_results, use_container_width=True)

            # Summary chart
            counts = df_results["Classification"].value_counts().reset_index()
            counts.columns = ["Classification", "Count"]
            fig = px.bar(
                counts,
                x="Classification",
                y="Count",
                color="Classification",
                title="Harmful vs neutral sequences",
                text="Count"
            )
            fig.update_layout(yaxis_title="Number of samples")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sequences detected. Please check your input format.")

# -----------------------------
# CRISPR-SHERLOCK design tab
# -----------------------------
with tab_crispr:
    st.subheader("Design a simple CRISPR-SHERLOCK assay")

    st.markdown(
        """
Paste a target DNA sequence (for example, a toxin gene fragment).
The app will:

- Propose a **20-nt CRISPR guide**
- Design simple **forward and reverse primers**
- Simulate an **assay signal** based on guide–target match
"""
    )

    target_default = (
        "ATGACCTGACCTGTTTACGATCGTAGCTAGCTAGCTAGCTAGCTGACTGACTGACTG"
    )

    target_seq = st.text_area(
        "Target DNA sequence",
        value=target_default,
        height=150
    )

    col1, col2 = st.columns(2)

    with col1:
        guide_length = st.slider("Guide length (nt)", 15, 24, 20)
    with col2:
        primer_length = st.slider("Primer length (nt)", 16, 24, 18)

    if st.button("Design CRISPR assay"):
        guide = design_crispr_guide(target_seq, guide_length)
        fwd, rev = design_primers(target_seq, primer_length)

        if not guide:
            st.error(
                "Could not design a guide. Ensure your sequence is long and contains only A/C/G/T."
            )
        else:
            st.markdown("#### Guide RNA")
            st.code(guide, language="text")

        if not fwd or not rev:
            st.warning(
                "Could not design primers — sequence may be too short. "
                "Try a longer input sequence or shorter primer length."
            )
        else:
            st.markdown("#### Primers")
            st.markdown("**Forward primer (5'→3')**")
            st.code(fwd, language="text")
            st.markdown("**Reverse primer (5'→3')**")
            st.code(rev, language="text")

        if guide:
            signal = simulate_sherlock_signal(guide, target_seq)
            st.markdown("#### Simulated SHERLOCK signal")
            st.write(f"Simulated fluorescence signal: **{signal}** arbitrary units (0–100)")

            fig_signal = px.bar(
                x=["Assay signal"],
                y=[signal],
                range_y=[0, 100],
                title="Simulated CRISPR-SHERLOCK assay signal",
                text=[signal]
            )
            fig_signal.update_layout(
                yaxis_title="Signal (a.u.)",
                xaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig_signal, use_container_width=True)

# -----------------------------
# Visualizations tab
# -----------------------------
with tab_visuals:
    st.subheader("Visualizing sample-level bloom risk")

    st.markdown(
        """
Generate a **synthetic dataset** of algae samples and see:

- Distribution of harmful vs neutral classifications
- Example "assay signal" distributions
"""
    )

    n_samples = st.slider("Number of synthetic samples", 10, 200, 40)

    if st.button("Generate synthetic dataset"):
        df_samples = generate_example_dataset(n_samples)
        st.dataframe(df_samples[["Sample ID", "Label"]], use_container_width=True)

        # Label distribution
        label_counts = df_samples["Label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]

        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(
                label_counts,
                names="Label",
                values="Count",
                title="Proportion of harmful vs neutral samples"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Simulate "assay signal" for each sample
            signals = []
            for _, row in df_samples.iterrows():
                label = row["Label"]
                # Higher signal for harmful, lower for neutral
                if label == "Harmful":
                    s = random.uniform(60, 100)
                elif label == "Neutral":
                    s = random.uniform(0, 40)
                else:
                    s = random.uniform(0, 20)
                signals.append(s)

            df_samples["Simulated signal"] = signals
            fig_hist = px.histogram(
                df_samples,
                x="Simulated signal",
                color="Label",
                nbins=20,
                title="Distribution of simulated CRISPR assay signals"
            )
            fig_hist.update_layout(xaxis_title="Signal (a.u.)")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown(
            """
**Interpretation (conceptual):**

- Samples classified as **harmful** tend to have **higher assay signal**.
- **Neutral** samples cluster at lower signal values.
- A real diagnostic system would calibrate thresholds based on
  experimental validation and controls.
"""
        )
