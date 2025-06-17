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
    linear_model_lib_pc1 = joblib.load(f"models/Linear-Regression-PC1-LibriSpeech-train-clean-100-17-speakers.pkl")
    linear_model_lib_pc4 = joblib.load(f"models/Linear-Regression-PC4-LibriSpeech-train-clean-100-17-speakers.pkl")
    # linear_model_pc29 = joblib.load(f"models/Linear-Regression-PC29-LibriSpeech-train-clean-100-17-speakers.pkl")

    linear_model_pc1 = joblib.load("/home/kyle/Projects/Streamlit-pitch-dashboard/models/Linear-Regression-PC1-VCTK-speakers.pkl")
    linear_model_pc4 = joblib.load("/home/kyle/Projects/Streamlit-pitch-dashboard/models/Linear-Regression-PC4-VCTK-speakers.pkl")
    rf_model = joblib.load(f"models/VCTK-random-forest-most-linear-pcs.pkl")
    # rf_model = joblib.load(f"models/VCTK-random-forest-most-linear-pcs-with-labels.pkl")
    # linear_model = joblib.load(f"models/Linear-Regression-LibriSpeech-train-clean-100-17-speakers-with-labels.pkl")
    return wavlm, hifigan, utt_pca, linear_model_pc1, linear_model_pc4, linear_model_lib_pc1, linear_model_lib_pc4, rf_model

# Use the cached function
wavlm, hifigan, utt_pca, lin_model_pc1, lin_model_pc4, lin_model_lib_pc1, lin_model_lib_pc4, rf_model = load_models()

st.title("Pitch control using Utterance-level PCA")

# Specify the directory
folder_path = Path('audio/')

# Get all files (not directories)
file_names = [f.name for f in sorted(folder_path.iterdir()) if f.is_file()]

clean = st.toggle(label="Clean Pitch", value=False)

st.markdown("### Audio file")
audio_file = st.selectbox(label="", options=file_names)
audio_path = f"{folder_path}/{audio_file}"
wav, sr = torchaudio.load(audio_path)
wav = wav.to(device)

# Feature extraction
with torch.inference_mode():
    x, _ = wavlm.extract_features(wav, output_layer=6)
    x_copy = x.squeeze(0).cpu().numpy()
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
try:
    st.markdown("## Original Audio")
    st.audio(audio_path)
    if clean:
        plt = plot_spectrogram_clean(audio_path, max_freq=1000, window_len=0.04)
        avg_original_pitch = compute_avg_cleaned_pitch(audio_path)
    else:
        plt = plot_spectrogram(audio_path, max_freq=1000, window_len=0.04)
        avg_original_pitch = compute_avg_pitch(audio_path)
    st.pyplot(plt)

    
    std_original_pitch = compute_std_pitch(audio_path)

    col1, col2 = st.columns(2)
    col1.metric(label="Average Pitch", value=f"{avg_original_pitch:.3f} Hz")
    col2.metric(label="Standard Deviation of Pitch", value=f"{std_original_pitch:.3f} Hz")

    # # Reset button
    # if st.button("Reset PC Values to Defaults"):
    #     st.session_state.pc1 = default_pc1
    #     st.session_state.pc4 = default_pc4
    #     st.session_state.pc29 = default_pc29

    # # Sliders
    # st.slider("PC 1 value", min_value=-30.0, max_value=30.0, key="pc1")
    # st.slider("PC 4 value", min_value=-30.0, max_value=30.0, key="pc4")
    # st.slider("PC 29 value", min_value=-30.0, max_value=30.0, key="pc29")

    st.markdown("## Adjust Principal Components")

    # col1, col2 = st.columns(2)

    if st.button("Reset PC Values to Defaults"):
        st.session_state.pc1 = default_pc1
        st.session_state.pc4 = default_pc4
        st.session_state.pc29 = default_pc29

    pc1_value = st.session_state.get("pc1", default_pc1)
    pc4_value = st.session_state.get("pc4", default_pc4)
    pc29_value = st.session_state.get("pc29", default_pc29)

    pc1 = st.slider("PC 1 value", min_value=-30.0, max_value=30.0, value=pc1_value, key="pc1_slider", format="%.2f")
    pc4 = st.slider("PC 4 value", min_value=-30.0, max_value=30.0, value=pc4_value, key="pc4_slider", format="%.2f")
    # st.slider("PC 1 value", min_value=-30.0, max_value=30.0, key="pc1", format="%.2f")
    # st.slider("PC 4 value", min_value=-30.0, max_value=30.0, key="pc4", format="%.2f")
    # st.slider("PC 29 value", min_value=-30.0, max_value=30.0, key="pc29", format="%.2f")

    # Apply button
    apply_changes = st.button("Apply PC Changes")

    if apply_changes:
        # Access values
        pc1 = st.session_state.pc1
        pc4 = st.session_state.pc4
        pc29 = st.session_state.pc29

        pc_dif1 = pc1 - compressed_x_mean[0, 0]
        pc_dif4 = pc4 - compressed_x_mean[0, 3]

        scaling = np.zeros(50)
        scaling[0] = pc_dif1
        scaling[3] = pc_dif4

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
        if clean:
            changed_avg_pitch = compute_avg_cleaned_pitch_numpy(file=wav_hat)
        else:
            changed_avg_pitch = compute_avg_pitch_numpy(file=wav_hat)
        changed_std_pitch = compute_std_pitch_numpy(file=wav_hat)

        col1, col2 = st.columns(2)
        col1.metric(label="Average Pitch of Changed Audio", value=f"{changed_avg_pitch:.3f} Hz")
        col2.metric(label="Standard Deviation of Pitch of Changed Audio", value=f"{changed_std_pitch:.3f} Hz")
        if clean:
            changed_plt = plot_spectrogram_numpy_clean(file=wav_hat, max_freq=1000, window_len=0.04)
        else:
            changed_plt = plot_spectrogram_numpy(file=wav_hat, max_freq=1000, window_len=0.04)
        st.pyplot(changed_plt)

    st.markdown("## PC prediction using VCTK Linear Regression")
    st.write("Note: The linear model was trained using all utterance data in the VCTK dataset")

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
    # target_pc29 = lin_model_pc29.predict(target_pitch_np)

    st.metric(label="Required PC 1 value", value=target_pc1)
    st.metric(label="Required PC 4 value", value=target_pc4)

    if st.button(label="Apply VCTK LR PC values"):
        st.session_state["pc1"] = float(target_pc1)
        st.session_state["pc4"] = float(target_pc4)
        st.rerun()

    st.markdown("## PC prediction using LibriSpeech Linear Regression")
    st.write("Note: The linear model was trained using the first 17 speakers of the LibriSpeech train-clean-100 dataset")
    
    target_lib_pc1 = lin_model_lib_pc1.predict(target_pitch_np)
    target_lib_pc4 = lin_model_lib_pc4.predict(target_pitch_np)
    # target_pc29 = lin_model_pc29.predict(target_pitch_np)

    st.metric(label="Required PC 1 value", value=target_lib_pc1)
    st.metric(label="Required PC 4 value", value=target_lib_pc4)

    if st.button(label="Apply LibriSpeech LR PC values"):
        st.session_state["pc1"] = float(target_lib_pc1)
        st.session_state["pc4"] = float(target_lib_pc4)
        st.rerun()
    # st.metric(label="Required PC 29 value", value=target_pc29)

    # RF with no gender label:

    st.markdown("## PC prediction using Random Forest")
    st.write("Note: The Random Forest was trained using the VCTK dataset, using an anchor-targets approach")
    
    current_pitch_float = float(avg_original_pitch)
    current_pitch_np = np.array([[current_pitch_float]])
    
    # st.write(current_pitch_np)
    # st.write(target_pitch_np)
    # st.write(compressed_x_mean)
    X = np.hstack([current_pitch_np, target_pitch_np, compressed_x_mean[:, [0, 3]]])

    y_hat = rf_model.predict(X)
    st.metric(label="Required PC 1 value", value=y_hat[:, 0])
    st.metric(label="Required PC 4 value", value=y_hat[:, 1])
    # st.metric(label="Required PC 29 value", value=y_hat[:, 2])

    if st.button(label="Apply RF PC values"):
        st.session_state.pc1 = float(y_hat[:, 0])
        st.session_state.pc4 = float(y_hat[:, 1])
        st.rerun()

    # RF with gender label:

    # st.markdown("## PC prediction using Random Forest")
    # st.write("Note: The Random Forest was trained using the VCTK dataset, using an anchor-targets approach")
    
    # current_pitch_float = float(avg_original_pitch)
    # current_pitch_np = np.array([[current_pitch_float]])
    # current_label = st.text_input("Please enter if the current speaker is male (0) or female (1). Only the number", value = "0.0")
    # current_label_float = float(current_label)
    # current_label_np = np.array([[current_label_float]])
    
    # # st.write(current_pitch_np)
    # # st.write(target_pitch_np)
    # # st.write(compressed_x_mean)
    # x = np.hstack([current_pitch_np, target_pitch_np, compressed_x_mean[:, 0:30], current_label_np])

    # y_hat = rf_model.predict(x)
    # st.metric(label="Required PC 1 value", value=y_hat[:, 0])
    # st.metric(label="Required PC 4 value", value=y_hat[:, 1])
    # # st.metric(label="Required PC 29 value", value=y_hat[:, 2])

    # Visualize original audio's PCs over time
    # st.markdown("## PC1 over time")
    # st.write(x_copy.shape)
    # converted_x = utt_pca.transform(x_copy)
    # pc1_plt = plot_pc_vs_time(x[:, 0], "1")
    # pc4_plt = plot_pc_vs_time(x[:, 3], "4")
    # st.pyplot(pc1_plt)
    # st.pyplot(pc4_plt)
except Exception as e:
    st.error(f"Error while loading spectrogram. It's a problem with the slider sensitivity, please reload and try not to move sliders for too long at a time :)")
    st.error(f"{e}")