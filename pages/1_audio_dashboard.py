import streamlit as st
import torch
import torch.nn.functional as F 
import torchaudio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
from helper_functions import compute_avg_pitch, compute_std_pitch, plot_spectrogram, plot_spectrogram_numpy, plot_spectrogram_clean, plot_spectrogram_numpy_clean
from helper_functions import compute_avg_pitch_numpy, compute_std_pitch_numpy, compute_avg_cleaned_pitch, compute_avg_cleaned_pitch_numpy, plot_pc_vs_time

st.set_page_config(layout="wide")

# Set the GPU to run all calculations when using torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device)
    utt_pca = joblib.load(f"models/utterance_level_train-100_pca_model_50_components.pkl")
    # Specify the directory
    folder_path = Path('/home/kyle/Projects/LibriSpeech/test-clean')
    file_paths = sorted(folder_path.rglob("*.flac"))
    return wavlm, hifigan, utt_pca, file_paths

# Use the cached function
wavlm, hifigan, utt_pca, file_paths = load_models()

st.title("Audio selection")

# Get all files (not directories)
file_names = [str(p) for p in file_paths]
selected = np.zeros(len(file_names))

# Set up session state for index
if "index" not in st.session_state:
    st.session_state.index = 0

# Set up session state for selected checkboxes
if "selected_files" not in st.session_state:
    st.session_state.selected_files = {}

# Get file paths
folder_path = Path('/home/kyle/Projects/LibriSpeech/test-clean')
file_paths = sorted(folder_path.rglob("*.flac"))
file_names = [str(p) for p in file_paths]
index = st.session_state.index
current_file = file_names[index]

# Initialize the checkbox state for the current file if not already set
if current_file not in st.session_state.selected_files:
    st.session_state.selected_files[current_file] = False

# UI
col1, col2, col3, col4 = st.columns([1, 4, 1, 1])
prev = col1.button("Previous")
col2.markdown(f"### {Path(current_file).stem}")
next = col3.button("Next")
check = col4.checkbox(
    "Select audio",
    value=st.session_state.selected_files[current_file],
    key=f"checkbox_{index}"
)

# Update checkbox state
st.session_state.selected_files[current_file] = check

# Navigation logic
if prev and index > 0:
    st.session_state.index -= 1
    st.rerun()
elif next and index < len(file_names) - 1:
    st.session_state.index += 1
    st.rerun()

wav, sr = torchaudio.load(file_paths[index])
wav = wav.to(device)
st.audio(file_paths[index])

plt = plot_spectrogram(f"{file_paths[index]}", max_freq=1000, window_len=0.04)
st.pyplot(plt)
avg_pitch = compute_avg_pitch(f"{file_paths[index]}")
std_pitch = compute_std_pitch(f"{file_paths[index]}")

col1, col2 = st.columns(2)
col1.metric(label="Average Pitch", value=f"{avg_pitch:.3f} Hz")
col2.metric(label="Standard Deviation of Pitch", value=f"{std_pitch:.3f} Hz")