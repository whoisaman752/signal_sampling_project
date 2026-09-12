import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Signal Lab", layout="wide")

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background: linear-gradient(45deg, #22c55e, #16a34a);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<h1 style='text-align: center;'>📡 Signal Sampling & Reconstruction </h1>", unsafe_allow_html=True)

# -------------------------------
# SESSION STATE
# -------------------------------
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.samplerate = None

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload WAV File", type=["wav"])

if uploaded_file:
    data, samplerate = sf.read(uploaded_file)

    if len(data.shape) > 1:
        data = data[:, 0]

    st.session_state.data = data
    st.session_state.samplerate = samplerate

# -------------------------------
# RESET
# -------------------------------
if st.session_state.data is not None:
    if st.button("🔄 Reset"):
        st.session_state.data = None
        st.session_state.samplerate = None
        st.rerun()

# -------------------------------
# MAIN LOGIC
# -------------------------------
if st.session_state.data is not None:

    data = st.session_state.data
    samplerate = st.session_state.samplerate

    # Sidebar
    st.sidebar.markdown("## 🎛 Controls")

    new_rate = st.sidebar.slider(
        "Sampling Rate (Hz)",
        min_value=1000,
        max_value=int(samplerate),
        value=int(samplerate // 2),
        step=1000
    )

    factor = max(int(samplerate / new_rate), 1)

    downsampled = data[::factor]
    reconstructed = np.repeat(downsampled, factor)[:len(data)]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Rate", f"{samplerate} Hz")
    col2.metric("New Rate", f"{new_rate} Hz")
    col3.metric("Compression Factor", f"{factor}x")

    # Audio
    st.markdown("## 🎧 Audio Comparison")

    choice = st.radio("Select Audio", ["Original", "Sampled", "Reconstructed"])

    if choice == "Original":
        st.audio(data, sample_rate=samplerate)
    elif choice == "Sampled":
        st.audio(downsampled, sample_rate=new_rate)
    else:
        st.audio(reconstructed, sample_rate=samplerate)

    # Aliasing
    if new_rate < samplerate / 2:
        st.error("⚠️ Aliasing Detected")
    else:
        st.success("✅ No Aliasing")

    # -------------------------------
    # VISUALIZATION (FIXED)
    # -------------------------------
    st.markdown("## 📊 Signal Visualization")

    t = np.linspace(0, len(data)/samplerate, len(data))
    sample_indices = np.arange(0, len(data), factor)

    limit = 2000

    fig, ax = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={'hspace': 0.6}
    )

    # Original
    ax[0].plot(t[:limit], data[:limit])
    ax[0].set_title("Original Signal")
    ax[0].grid(True)

    # Sampled
    ax[1].plot(t[:limit], data[:limit], alpha=0.3)
    ax[1].scatter(t[sample_indices][:200], downsampled[:200])
    ax[1].set_title("Sampled Signal")
    ax[1].grid(True)

    # Reconstructed
    ax[2].plot(t[:limit], reconstructed[:limit])
    ax[2].set_title("Reconstructed Signal")
    ax[2].grid(True)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

    # -------------------------------
    # ERROR
    # -------------------------------
    error = np.mean((data - reconstructed) ** 2)
    st.metric("MSE", f"{error:.6f}")

    # -------------------------------
    # FFT
    # -------------------------------
    st.markdown("## 📡 Frequency Spectrum")

    def plot_fft(signal_data, sr, title):
        fft = np.fft.fft(signal_data)
        freq = np.fft.fftfreq(len(fft), 1/sr)

        fig, ax = plt.subplots()
        ax.plot(freq[:len(freq)//2], np.abs(fft[:len(freq)//2]))
        ax.set_title(title)
        ax.grid(True)
        return fig

    col1, col2 = st.columns(2)
    col1.pyplot(plot_fft(data, samplerate, "Original Spectrum"))
    col2.pyplot(plot_fft(reconstructed, samplerate, "Reconstructed Spectrum"))