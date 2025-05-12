import torch
import torch.nn.functional as F 
import torchaudio
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
from helper_functions import compute_avg_pitch, compute_std_pitch, plot_spectrogram, plot_spectrogram_numpy, compute_avg_pitch_numpy, compute_std_pitch_numpy

st.set_page_config(layout="wide")

# Set the GPU to run all calculations when using torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device)
    utt_pca = joblib.load(f"models/utterance_level_train-100_pca_model_50_components.pkl")
    linear_model_pc1 = joblib.load(f"models/Linear-Regression-PC1-LibriSpeech-train-clean-100-17-speakers.pkl")
    linear_model_pc4 = joblib.load(f"models/Linear-Regression-PC4-LibriSpeech-train-clean-100-17-speakers.pkl")
    linear_model_pc29 = joblib.load(f"models/Linear-Regression-PC29-LibriSpeech-train-clean-100-17-speakers.pkl")
    # linear_model = joblib.load(f"models/Linear-Regression-LibriSpeech-train-clean-100-17-speakers-with-labels.pkl")
    return wavlm, hifigan, utt_pca, linear_model_pc1, linear_model_pc4, linear_model_pc29

# Use the cached function
wavlm, hifigan, utt_pca, lin_model_pc1, lin_model_pc4, lin_model_pc29 = load_models()

st.title("Pitch control using PCA")

# Specify the directory
folder_path = Path('audio/')

# Get all files (not directories)
file_names = [f.name for f in sorted(folder_path.iterdir()) if f.is_file()]

st.markdown("### Audio file")
audio_file = st.selectbox(label="", options=file_names)
audio_path = f"{folder_path}/{audio_file}"
wav, sr = torchaudio.load(audio_path)
wav = wav.to(device)

# Feature extraction
with torch.inference_mode():
    x, _ = wavlm.extract_features(wav, output_layer=6)
    x_mean = x.mean(dim=1)
    x_mean = x_mean.cpu().numpy()

compressed_x_mean = utt_pca.transform(x_mean)

# Defaults for PCs 1, 4, and 29
default_pc1 = float(compressed_x_mean[0, 0])
default_pc4 = float(compressed_x_mean[0, 3])
default_pc29 = float(compressed_x_mean[0, 28])

# Track audio changes
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = audio_file

# Initialize session state
for key, default in zip(["pc1", "pc4", "pc29"], [default_pc1, default_pc4, default_pc29]):
    if key not in st.session_state:
        st.session_state[key] = default

# Reset if audio changed
if st.session_state.last_audio_file != audio_file:
    st.session_state.pc1 = default_pc1
    st.session_state.pc4 = default_pc4
    st.session_state.pc29 = default_pc29
    st.session_state.last_audio_file = audio_file


# UI elements
st.markdown("## Original Audio")
st.audio(audio_path)

plt = plot_spectrogram(audio_path, max_freq=1000, window_len=0.04)
st.pyplot(plt)

avg_pitch = compute_avg_pitch(audio_path)
std_pitch = compute_std_pitch(audio_path)

col1, col2 = st.columns(2)
col1.metric(label="Average Pitch", value=f"{avg_pitch:.3f} Hz")
col2.metric(label="Standard Deviation of Pitch", value=f"{std_pitch:.3f} Hz")

# Reset button
if st.button("Reset PC Values to Defaults"):
    st.session_state.pc1 = default_pc1
    st.session_state.pc4 = default_pc4
    st.session_state.pc29 = default_pc29

# Sliders
st.slider("PC 1 value", min_value=-50.0, max_value=50.0, key="pc1")
st.slider("PC 4 value", min_value=-50.0, max_value=50.0, key="pc4")
st.slider("PC 29 value", min_value=-50.0, max_value=50.0, key="pc29")

# Access values
pc1 = st.session_state.pc1
pc4 = st.session_state.pc4
pc29 = st.session_state.pc29

pc_dif1 = pc1 - compressed_x_mean[0, 0]
pc_dif4 = pc4 - compressed_x_mean[0, 3]
pc_dif29 = pc29 - compressed_x_mean[0, 28]

scaling = np.zeros(50)
scaling[0] = pc_dif1
scaling[3] = pc_dif4
scaling[28] = pc_dif29

# Cast scaling back up to normal dimension so that it can be added to audio frames to scale them in the higher dimension:
uncompressed_scaling = utt_pca.inverse_transform([scaling]) - utt_pca.mean_ #Result of shape (1, 1024)
# st.write(f"{uncompressed_scaling.shape}")
uncompressed_scaling = torch.from_numpy(uncompressed_scaling).float().to(device)

# Broadcast across all frames
delta = uncompressed_scaling.expand(x.shape[0], -1)  # Shape: (T, 1024) where T is the number of frames in x
# st.write(f"{delta.shape}")

# Apply change
x_modified = x + delta  # Shape: (T, 1024)
# st.write(f"{x_modified.shape}")

# Generate new waveform
with torch.inference_mode():
    wav_hat = hifigan(x_modified).squeeze().cpu().numpy()

st.markdown("## Changed Audio")
st.audio(wav_hat, sample_rate=16000)

changed_avg_pitch = compute_avg_pitch_numpy(file=wav_hat)
changed_std_pitch = compute_std_pitch_numpy(file=wav_hat)

col1, col2 = st.columns(2)
col1.metric(label="Average Pitch of Changed Audio", value=f"{changed_avg_pitch:.3f} Hz")
col2.metric(label="Standard Deviation of Pitch of Changed Audio", value=f"{changed_std_pitch:.3f} Hz")

changed_plt = plot_spectrogram_numpy(file=wav_hat, max_freq=1000, window_len=0.04)
st.pyplot(changed_plt)

st.markdown("## PC prediction using Linear Regression")
st.write("Note: The linear model was trained using only the first 17 speakers of the LibriSpeech train-clean-100 dataset")

# col1, col2 = st.columns(2)
# target_pitch = col1.text_input("Please enter desired target Pitch (Hz value only)", value="0.0")
# current_label = col2.text_input("Please enter if the current speaker is male (0) or female (1). Only the numeber", value = "0.0")
# target_pitch_float = float(target_pitch)
# current_label_float = float(current_label)
# target_pitch_np = np.array([[target_pitch_float]])
# current_label_np = np.array([[current_label_float]])
# x_value = np.hstack([target_pitch_np, current_label_np])

# target_pc = linear_model.predict(x_value)
# st.metric(label="Required PC 1 value", value=target_pc)
    
target_pitch = st.text_input("Please enter desired target Pitch (Hz value only)", value="80.0")
target_pitch_float = float(target_pitch)
target_pitch_np = np.array([[target_pitch_float]])
target_pc1 = lin_model_pc1.predict(target_pitch_np)
target_pc4 = lin_model_pc4.predict(target_pitch_np)
target_pc29 = lin_model_pc29.predict(target_pitch_np)

st.metric(label="Required PC 1 value", value=target_pc1)
st.metric(label="Required PC 4 value", value=target_pc4)
st.metric(label="Required PC 29 value", value=target_pc29)