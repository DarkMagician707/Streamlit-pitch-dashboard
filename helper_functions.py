import parselmouth
from parselmouth.praat import call
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F 
import torchaudio
from torchaudio.transforms import Resample
import os
from scipy.stats import linregress
import scipy.interpolate
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

def parse_speaker_info(speaker_file):
    """
    Parses the speaker-info.txt file to extract speaker genders.
    
    Args:
        speaker_file (str): Path to the speaker-info.txt file.
    
    Returns:
        dict: A dictionary mapping speaker IDs (as strings, e.g., 'p225') to their genders ('M' or 'F').
    """
    speaker_gender = {}

    with open(speaker_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        if len(parts) >= 3:
            speaker_id = f"p{parts[0]}"  # Prefix with 'p'
            gender = parts[2]
            if gender in {'M', 'F'}:
                speaker_gender[speaker_id] = gender

    return speaker_gender

def parse_speaker_info_lib(speaker_file):
    """
    Parses the SPEAKERS.TXT file to extract speaker genders.
    
    Args:
        speaker_file (str): Path to the SPEAKERS.TXT file.
    
    Returns:
        dict: A dictionary mapping speaker IDs to their genders.
    """
    speaker_gender = {}
    
    with open(speaker_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(";") or line.strip() == "":
                continue  # Skip comments and empty lines
            
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue  # Skip malformed lines
            
            speaker_id = parts[0].strip()
            gender = parts[1].strip()
            
            if speaker_id.isdigit() and gender in {'M', 'F'}:
                speaker_gender[speaker_id] = gender
    
    return speaker_gender


def get_speaker_utterances(data_path, speaker_file, num_speakers=None):
    """
    Finds all utterance file paths for a given number of VCTK speakers, including their gender.
    
    Args:
        data_path (str): Path to the VCTK 'wav48' directory.
        speaker_file (str): Path to the speaker-info.txt file.
        num_speakers (int, optional): Number of speakers to include.
    
    Returns:
        dict: A dictionary where keys are speaker IDs (e.g., 'p225'), and values are:
              {'gender': 'M' or 'F', 'utterances': [list of .wav file paths]}
    """
    # Step 1: Load speaker gender info
    speaker_gender = parse_speaker_info(speaker_file)

    # Step 2: List all available speaker folders (e.g., p225, p226)
    speakers = sorted(os.listdir(data_path), key=lambda x: int(x.lstrip('p')))
    if num_speakers is not None:
        speakers = speakers[:num_speakers]

    speaker_data = {}

    for speaker in speakers:
        speaker_path = os.path.join(data_path, speaker)
        if not os.path.isdir(speaker_path):
            continue

        gender = speaker_gender.get(speaker, "Unknown")

        # Get all .wav files in the speaker's folder
        utterance_files = [
            os.path.join(speaker_path, file)
            for file in os.listdir(speaker_path)
            if file.endswith(".wav")
        ]

        if utterance_files:
            speaker_data[speaker] = {
                "gender": gender,
                "utterances": sorted(utterance_files)
            }

    return speaker_data

def get_speaker_utterances_lib(data_path, speaker_file, num_speakers=None):
    """
    Finds all utterance file paths for a given number of speakers, including their gender.
    
    Args:
        data_path (str): Path to the "train-clean-100" directory.
        speaker_file (str): Path to the SPEAKERS.TXT file.
        num_speakers (int, optional): Number of speakers to include.
    
    Returns:
        dict: A dictionary where keys are speaker IDs, and values are:
              {'gender': 'M' or 'F', 'utterances': [list of file paths]}
    """
    # Step 1: Load speaker gender info
    speaker_gender = parse_speaker_info_lib(speaker_file)
    
    # Step 2: List all available speakers in the dataset
    speakers = sorted(os.listdir(data_path), key=lambda x: int(x))  # Sorted for consistency
    # print(speakers)
    if num_speakers is not None:
        speakers = speakers[:num_speakers]  # Select the first N speakers

    speaker_data = {}

    for speaker in speakers:
        speaker_path = os.path.join(data_path, speaker)
        if not os.path.isdir(speaker_path):  # Ensure it's a directory
            continue
        
        # Get gender info (default to 'Unknown' if not found)
        gender = speaker_gender.get(speaker, "Unknown")
        
        utterance_files = []
        
        # Each speaker has multiple chapters
        for chapter in sorted(os.listdir(speaker_path), key=lambda x: int(x)):
            # print(chapter)
            chapter_path = os.path.join(speaker_path, chapter)
            if not os.path.isdir(chapter_path):
                continue
            
            # Find all .flac files (utterances)
            for file in os.listdir(chapter_path):
                if file.endswith(".flac"):
                    utterance_files.append(os.path.join(chapter_path, file))
        # print(sorted(utterance_files))
        if utterance_files:
            speaker_data[speaker] = {
                "gender": gender,
                "utterances": sorted(utterance_files)  # Sorted for consistency
            }
    
    return speaker_data


def compute_avg_pitch(file_path, pitch_floor = 70, pitch_ceiling = 400):
    sound = parselmouth.Sound(file_path)
    
    # Extract pitch using Praat's method
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    # pitch = sound.to_pitch(pitch_floor = 50, pitch_ceiling = 800)
    
    # Get pitch values (ignoring unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    # print(pitch_values)
    # pitch_values[pitch_values > 400] = 0 # Filter out random outliers that Parselmouth detects
    pitch_values = pitch_values[pitch_values > 0]  # Remove zero (unvoiced parts)
    # print(pitch_values)
    
    if len(pitch_values) == 0:
        return None  # No voiced frames detected
    
    return np.mean(pitch_values)

def compute_avg_cleaned_pitch(file_path, pitch_floor = 70, pitch_ceiling = 400):
    sound = parselmouth.Sound(file_path)
    
    # Extract pitch using Praat's method
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    # pitch = sound.to_pitch(pitch_floor = 50, pitch_ceiling = 800)
    
    # Get pitch values (ignoring unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    # print(pitch_values)
    # pitch_values[pitch_values > 400] = 0 # Filter out random outliers that Parselmouth detects
    pitch_values = pitch_values[pitch_values > 0]  # Remove zero (unvoiced parts)
    # print(pitch_values)
    cleaned_pitch, outlier_mask = detect_and_handle_pitch_outliers(pitch_values)

    if len(cleaned_pitch) == 0:
        return None  # No voiced frames detected
    
    return np.mean(cleaned_pitch)

def compute_avg_pitch_numpy(file, sr = 16000, pitch_floor = 70, pitch_ceiling = 400):
    sound = parselmouth.Sound(values = file, sampling_frequency = sr)
    
    # Extract pitch using Praat's method
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    # pitch = sound.to_pitch(pitch_floor = 50, pitch_ceiling = 800)
    
    # Get pitch values (ignoring unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    # print(pitch_values)
    # pitch_values[pitch_values > 400] = 0 # Filter out random outliers that Parselmouth detects
    pitch_values = pitch_values[pitch_values > 0]  # Remove zero (unvoiced parts)
    # print(pitch_values)
    
    if len(pitch_values) == 0:
        return None  # No voiced frames detected
    
    return np.mean(pitch_values)

def compute_avg_cleaned_pitch_numpy(file, sr = 16000, pitch_floor = 70, pitch_ceiling = 400):
    sound = parselmouth.Sound(values = file, sampling_frequency = sr)
    
    # Extract pitch using Praat's method
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    # pitch = sound.to_pitch(pitch_floor = 50, pitch_ceiling = 800)
    
    # Get pitch values (ignoring unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    # print(pitch_values)
    # pitch_values[pitch_values > 400] = 0 # Filter out random outliers that Parselmouth detects
    pitch_values = pitch_values[pitch_values > 0]  # Remove zero (unvoiced parts)
    # print(pitch_values)
    cleaned_pitch, outlier_mask = detect_and_handle_pitch_outliers(pitch_values)

    if len(cleaned_pitch) == 0:
        return None  # No voiced frames detected
    
    return np.mean(cleaned_pitch)

def compute_std_pitch(file_path, pitch_floor = 70, pitch_ceiling = 400):
    sound = parselmouth.Sound(file_path)
    
    # Extract pitch using Praat's method
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    # pitch = sound.to_pitch(pitch_floor = 50, pitch_ceiling = 800)
    
    # Get pitch values (ignoring unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    # print(pitch_values)
    # pitch_values[pitch_values > 400] = 0 # Filter out random outliers that Parselmouth detects
    # pitch_values = pitch_values[pitch_values > 0]  # Remove zero (unvoiced parts)
    # print(pitch_values)
    
    if len(pitch_values) == 0:
        return None  # No voiced frames detected
    
    return np.std(pitch_values)

def compute_std_pitch_numpy(file, sr = 16000, pitch_floor = 70, pitch_ceiling = 400):
    sound = parselmouth.Sound(values = file, sampling_frequency = sr)
    
    # Extract pitch using Praat's method
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    # pitch = sound.to_pitch(pitch_floor = 50, pitch_ceiling = 800)
    
    # Get pitch values (ignoring unvoiced frames)
    pitch_values = pitch.selected_array['frequency']
    # print(pitch_values)
    # pitch_values[pitch_values > 400] = 0 # Filter out random outliers that Parselmouth detects
    # pitch_values = pitch_values[pitch_values > 0]  # Remove zero (unvoiced parts)
    # print(pitch_values)
    
    if len(pitch_values) == 0:
        return None  # No voiced frames detected
    
    return np.std(pitch_values)

def compute_avg_intensity(file_path):
    sound = parselmouth.Sound(file_path)
    
    # Extract intensity using Praat's method
    # intensity = sound.to_intensity(subtract_mean=False)
    intensity = sound.to_intensity()
    
    # Get intensity values
    intensity_values = intensity.values
    intensity_values = intensity_values[intensity_values > 0]  # Remove zero (unvoiced parts)

    if len(intensity_values) == 0:
        return None  # No voiced frames detected
    
    return np.mean(intensity_values)

def compute_std_intensity(file_path):
    sound = parselmouth.Sound(file_path)
    
    # Extract intensity using Praat's method
    # intensity = sound.to_intensity(subtract_mean=False)
    intensity = sound.to_intensity()
    
    # Get intensity values
    intensity_values = intensity.values
    intensity_values = intensity_values[intensity_values > 0]  # Remove zero (unvoiced parts)

    if len(intensity_values) == 0:
        return None  # No voiced frames detected
    
    return np.std(intensity_values)

def compute_avg_intensity_numpy(file, sr = 16000):
    sound = parselmouth.Sound(values = file, sampling_frequency = sr)
    
    # Extract intensity using Praat's method
    # intensity = sound.to_intensity(subtract_mean=False)
    intensity = sound.to_intensity()
    
    # Get intensity values
    intensity_values = intensity.values
    intensity_values = intensity_values[intensity_values > 0]  # Remove zero (unvoiced parts)

    if len(intensity_values) == 0:
        return None  # No voiced frames detected
    
    return np.mean(intensity_values)

def compute_std_intensity_numpy(file, sr = 16000):
    sound = parselmouth.Sound(values = file, sampling_frequency = sr)
    
    # Extract intensity using Praat's method
    # intensity = sound.to_intensity(subtract_mean=False)
    intensity = sound.to_intensity()
    
    # Get intensity values
    intensity_values = intensity.values
    intensity_values = intensity_values[intensity_values > 0]  # Remove zero (unvoiced parts)

    if len(intensity_values) == 0:
        return None  # No voiced frames detected
    
    return np.std(intensity_values)

def plot_pitch_vs_pc(utt_compressed, average_pitch, gender_labels, hertz_or_semi = "hertz"):    
# Assign colors based on gender
    colors = ["blue" if g == "M" else "red" for g in gender_labels]

    # Number of principal components to plot
    num_plots = 12

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 2 rows, 5 columns
    axes = axes.flatten()  # Flatten to easily index axes

    for i in range(0, num_plots):
        pc_values = utt_compressed[:, i]  # Extract principal component `i`
        
        # Fit a trend line if correlation is significant
        slope, intercept, r_value, p_value, std_err = linregress(average_pitch.squeeze(), pc_values)

        # Scatter plot
        ax = axes[i]
        ax.scatter(average_pitch, pc_values, c=colors, alpha=0.7)

        # Only plot trend line if r-value (correlation coefficient) is larger than 0.7 (absolute of r-value) (since this shows a strong linear relationship)
        if abs(r_value) >= 0.4:
            x_vals = np.linspace(min(average_pitch), max(average_pitch), 100)
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color="black", linestyle="--", label=f"Trend (r={r_value:.2f})")

        # Labels and title
        if hertz_or_semi == "hertz":
            ax.set_xlabel("Average Pitch (Hz)")
        else:
            ax.set_xlabel("Average Pitch (Semitones)")
        ax.set_ylabel(f"PC {i+1} Value")
        ax.set_title(f"Principal Component {i+1} vs Pitch")
        # ax.legend(["Trend Line" if p_value < 0.05 else None, "Male", "Female"], loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Adjust layout
    # plt.title("Graph showing Principal Components vs Average Utterance Pitch for Two Speakers")
    plt.tight_layout()
    plt.show()

def plot_top_linear_feature_vs_pc(utt_compressed, feature, gender_labels, label, threshold = 0.4):    
    # Assign colors based on gender
    colors = ["blue" if g == "M" else "red" for g in gender_labels]

    lin_pc = []
    lin_r_value = []
    for i in range(0, 50):
        pc_values = utt_compressed[:, i]  # Extract principal component `i`
            
        # Fit a trend line if correlation is significant
        slope, intercept, r_value, p_value, std_err = linregress(feature.squeeze(), pc_values)
        
        if abs(r_value) >= threshold:
            lin_pc.append(i)
            lin_r_value.append(r_value)

    # Number of principal components to plot
    num_plots = len(lin_pc)

    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 2 rows, 5 columns
    axes = axes.flatten()  # Flatten to easily index axes

    for i in range(0, num_plots):
        pc_values = utt_compressed[:, lin_pc[i]]  # Extract principal component `i`
        
        # Fit a trend line if correlation is significant
        slope, intercept, r_value, p_value, std_err = linregress(feature.squeeze(), pc_values)

        # Scatter plot
        ax = axes[i]
        ax.scatter(feature, pc_values, c=colors, alpha=0.7)
        
        x_vals = np.linspace(min(feature), max(feature), 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color="black", linestyle="--", label=f"Trend (r={r_value:.2f})")

        # Labels and title
        ax.set_xlabel(f"Average {label}")
        ax.set_ylabel(f"PC {lin_pc[i]+1} Value")
        ax.set_title(f"Principal Component {lin_pc[i]+1} vs {label}")
        # ax.legend(["Trend Line" if p_value < 0.05 else None, "Male", "Female"], loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Adjust layout
    # plt.title("Graph showing Principal Components vs Average Utterance Pitch for Two Speakers")
    plt.tight_layout()
    plt.show()

def resample_to_16k(input_path, output_path, orig_sr=48000, target_sr=16000):
    """
    Resamples all .wav files from 48kHz to 16kHz using torchaudio.

    Args:
        input_path (str): Root directory of the VCTK corpus (e.g., 'wav48').
        output_path (str): Directory to save 16kHz audio (will mirror structure).
        orig_sr (int): Original sample rate (default: 48,000).
        target_sr (int): Target sample rate (default: 16,000).
    """
    os.makedirs(output_path, exist_ok=True)

    speakers = [s for s in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, s))]

    for speaker in speakers:
        speaker_dir = os.path.join(input_path, speaker)
        target_dir = os.path.join(output_path, speaker)
        os.makedirs(target_dir, exist_ok=True)

        for wav_file in os.listdir(speaker_dir):
            if wav_file.endswith('.wav'):
                src_path = os.path.join(speaker_dir, wav_file)
                dst_path = os.path.join(target_dir, wav_file)

                # Load audio
                waveform, sr = torchaudio.load(src_path)

                if sr != target_sr:
                    resampler = Resample(orig_freq=sr, new_freq=target_sr)
                    waveform = resampler(waveform)

                torchaudio.save(dst_path, waveform, target_sr)
                print(f"Saved: {dst_path}")

def get_anchor_target_pairs(anchor_target_path):
    
    # Read and parse the file line by line
    with open(anchor_target_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Create a list to hold (anchor, [targets]) pairs
    anchor_target_pairs = []
    current_anchor = None
    current_targets = []

    for line in lines:
        if line.startswith("Anchor:"):
            # Save the previous anchor-target group if one was being collected
            if current_anchor and current_targets:
                anchor_target_pairs.append((current_anchor, current_targets))
            # Start a new group
            current_anchor = line.replace("Anchor:", "").strip()
            current_targets = []
        elif line.startswith("➤ Target:"):
            target = line.replace("➤ Target:", "").strip()
            current_targets.append(target)

    # Don't forget to add the last group
    if current_anchor and current_targets:
        anchor_target_pairs.append((current_anchor, current_targets))

    return anchor_target_pairs

def create_training_data(anchor_target_pairs, audio_path):
    anchor_stacked_mean_features = []
    anchor_stacked_pitches = []
    target_stacked_mean_features = []
    target_stacked_pitches = []

    for anchor_id, target_ids in anchor_target_pairs:
        count = 0
        for target_id in target_ids:
            target_mean_features = np.load(f"{audio_path}/{target_id.split('_')[0]}/{target_id}-average-wavlm-features.npy")
            target_pitch = np.load(f"{audio_path}/{target_id.split('_')[0]}/{target_id}-average_pitch.npy")
            
            target_stacked_mean_features.append(target_mean_features)
            target_stacked_pitches.append(target_pitch)
            count = count + 1
        
        anchor_mean_features = np.load(f"{audio_path}/{anchor_id.split('_')[0]}/{anchor_id}-average-wavlm-features.npy")
        anchor_pitch = np.load(f"{audio_path}/{anchor_id.split('_')[0]}/{anchor_id}-average_pitch.npy")

        anchor_stacked_mean_features.extend([anchor_mean_features] * count)
        anchor_stacked_pitches.extend([anchor_pitch] * count)
    
    anchor_stacked_mean_features = np.vstack(anchor_stacked_mean_features)
    anchor_stacked_pitches = np.vstack(anchor_stacked_pitches)
    target_stacked_mean_features = np.vstack(target_stacked_mean_features)
    target_stacked_pitches = np.vstack(target_stacked_pitches)

    return anchor_stacked_mean_features, anchor_stacked_pitches, target_stacked_mean_features, target_stacked_pitches

def plot_waveform(path):
    sns.set_theme() # Use seaborn's default style to make attractive graphs

    # Plot nice figure for low pitch using Python's "standard" matplotlib library
    snd = parselmouth.Sound(path)
    plt.figure(figsize= (18, 5))
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform of audio file")
    plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

def draw_pc(pcs, spectrogram):

    plt.plot(spectrogram.xs(), pcs, 'o', markersize=5, color='w', label="Pitch")
    plt.plot(spectrogram.xs(), pcs, 'o', markersize=2, color='blue')

    plt.ylim([0, spectrogram.ymax])  # Ensure pitch range matches spectrogram range
    plt.ylabel("fundamental frequency [Hz]")

def draw_pitch(pitch, spectrogram):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan  # Remove unvoiced samples
    pitch_values[pitch_values > 400] = np.nan  # Filter out high values

    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w', label="Pitch")
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2, color='red')

    plt.ylim([0, spectrogram.ymax])  # Ensure pitch range matches spectrogram range
    plt.ylabel("fundamental frequency [Hz]")

def draw_clean_pitch(pitch, spectrogram):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan  # Remove unvoiced samples
    pitch_values[pitch_values > 400] = np.nan  # Filter out high values

    cleaned_pitch, outlier_mask = detect_and_handle_pitch_outliers(pitch_values)


    plt.plot(pitch.xs(), cleaned_pitch, 'o', markersize=5, color='w', label="Pitch")
    plt.plot(pitch.xs(), cleaned_pitch, 'o', markersize=2, color='red')

    plt.ylim([0, spectrogram.ymax])  # Ensure pitch range matches spectrogram range
    plt.ylabel("fundamental frequency [Hz]")

def plot_spectrogram(path, max_freq = 2000, window_len = 0.03, dynamic_range=70, pitch=None):
    snd = parselmouth.Sound(path)
    spectrogram = snd.to_spectrogram(window_length = window_len, maximum_frequency = max_freq)

    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)

    plt.figure(figsize=(20, 7))
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='inferno_r', shading='auto')
    
    # Restrict frequency range to match pitch
    if pitch:
        plt.ylim([0, pitch.ceiling])
    else:
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
    # else:
    #     plt.ylim([spectrogram.ymin, 2000])

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    pitch = snd.to_pitch(pitch_floor = 70, pitch_ceiling = 400)

    # Plot pitch contour
    plt.twinx()
    draw_pitch(pitch, spectrogram=spectrogram)

    plt.xlim([snd.xmin, snd.xmax])
    plt.title("Spectrogram with Aligned Pitch Contour")
    return plt
    # plt.show() # or plt.savefig("spectrogram_0.03.pdf")

def plot_spectrogram_clean(path, max_freq = 2000, window_len = 0.03, dynamic_range=70, pitch=None):
    snd = parselmouth.Sound(path)
    spectrogram = snd.to_spectrogram(window_length = window_len, maximum_frequency = max_freq)

    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)

    plt.figure(figsize=(20, 7))
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='inferno_r', shading='auto')
    
    # Restrict frequency range to match pitch
    if pitch:
        plt.ylim([0, pitch.ceiling])
    else:
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
    # else:
    #     plt.ylim([spectrogram.ymin, 2000])

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    pitch = snd.to_pitch(pitch_floor = 70, pitch_ceiling = 400)

    # Plot pitch contour
    plt.twinx()
    draw_clean_pitch(pitch, spectrogram=spectrogram)

    plt.xlim([snd.xmin, snd.xmax])
    plt.title("Spectrogram with Aligned Pitch Contour")
    return plt
    # plt.show() # or plt.savefig("spectrogram_0.03.pdf")

def plot_spectrogram_numpy(file, sr = 16000, max_freq = 2000, window_len = 0.03, dynamic_range=70, pitch=None):
    snd = parselmouth.Sound(values = file, sampling_frequency = sr)
    spectrogram = snd.to_spectrogram(window_length = window_len, maximum_frequency = max_freq)

    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)

    plt.figure(figsize=(20, 7))
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='inferno_r', shading='auto')
    
    # Restrict frequency range to match pitch
    if pitch:
        plt.ylim([0, pitch.ceiling])
    else:
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
    # else:
    #     plt.ylim([spectrogram.ymin, 2000])

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    pitch = snd.to_pitch(pitch_floor = 70, pitch_ceiling = 400)

    # Plot pitch contour
    plt.twinx()
    draw_pitch(pitch, spectrogram=spectrogram)

    plt.xlim([snd.xmin, snd.xmax])
    plt.title("Spectrogram with Aligned Pitch Contour")
    return plt

def plot_spectrogram_numpy_clean(file, sr = 16000, max_freq = 2000, window_len = 0.03, dynamic_range=70, pitch=None):
    snd = parselmouth.Sound(values = file, sampling_frequency = sr)
    spectrogram = snd.to_spectrogram(window_length = window_len, maximum_frequency = max_freq)

    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)

    plt.figure(figsize=(20, 7))
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='inferno_r', shading='auto')
    
    # Restrict frequency range to match pitch
    if pitch:
        plt.ylim([0, pitch.ceiling])
    else:
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
    # else:
    #     plt.ylim([spectrogram.ymin, 2000])

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    pitch = snd.to_pitch(pitch_floor = 70, pitch_ceiling = 400)

    # Plot pitch contour
    plt.twinx()
    draw_clean_pitch(pitch, spectrogram=spectrogram)

    plt.xlim([snd.xmin, snd.xmax])
    plt.title("Spectrogram with Aligned Pitch Contour")
    return plt

def plot_pc_vs_time(pcs, label):
    # Define frame shift (in seconds)
    frame_shift = 0.02  # WavLM uses 20 ms frames for audio

    # Time axis
    time_axis = np.arange(len(pcs)) * frame_shift

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, pcs, marker='o', linestyle='-')
    plt.title(f"PC {label} Values Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(f"PC {label}Value")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    return plt

def detect_and_handle_pitch_outliers(pitch_values):
    pitch_values = np.array(pitch_values)
    
    # Identify voiced frames: non-zero, non-nan
    is_voiced = (pitch_values > 0) & (~np.isnan(pitch_values))
    voiced_values = pitch_values[is_voiced]

    if len(voiced_values) < 2:
        # Not enough data to clean
        return pitch_values.copy(), np.zeros_like(pitch_values, dtype=bool)

    # Detect outliers using IQR
    Q1 = np.percentile(voiced_values, 25)
    Q3 = np.percentile(voiced_values, 75)
    IQR = Q3 - Q1
    threshold = 2.5
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Outlier mask (only on voiced frames)
    outlier_mask = np.zeros_like(pitch_values, dtype=bool)
    outlier_mask[is_voiced] = (pitch_values[is_voiced] < lower_bound) | (pitch_values[is_voiced] > upper_bound)

    # Create new array with cleaned values
    cleaned_pitch = pitch_values.copy()

    # Indices of non-outlier voiced frames (valid for interpolation)
    valid_voiced_indices = np.where(is_voiced & ~outlier_mask)[0]
    valid_voiced_values = pitch_values[valid_voiced_indices]

    # Indices of outlier voiced frames to interpolate
    outlier_voiced_indices = np.where(outlier_mask)[0]

    if len(valid_voiced_indices) >= 2:
        interp_func = scipy.interpolate.interp1d(
            valid_voiced_indices,
            valid_voiced_values,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        cleaned_pitch[outlier_voiced_indices] = interp_func(outlier_voiced_indices)
    else:
        # Fallback to median if not enough valid points
        fallback_value = np.median(voiced_values)
        cleaned_pitch[outlier_voiced_indices] = fallback_value

    # Unvoiced values are preserved
    return cleaned_pitch, outlier_mask

def get_jitter(audio_file, pitch_floor = 70, pitch_ceiling = 400):
    snd = parselmouth.Sound(audio_file)
    pointProcess = call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    local_jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_abs_jitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    return local_jitter, local_abs_jitter, rap_jitter, ppq5_jitter, ddp_jitter

def get_shimmer(audio_file, pitch_floor = 70, pitch_ceiling = 400):
    snd = parselmouth.Sound(audio_file)
    pointProcess = call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    local_shimmer =  call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    local_shimmer_db = call([snd, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([snd, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call([snd, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer =  call([snd, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([snd, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return local_shimmer, local_shimmer_db, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer

def get_hnr(audio_file, pitch_floor = 70, pitch_ceiling = 400):
    snd = parselmouth.Sound(audio_file)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    return hnr, harmonicity

def get_jitter_numpy(file, sr = 16000, pitch_floor = 70, pitch_ceiling = 400):
    snd = parselmouth.Sound(values = file, sampling_frequency = sr)
    pointProcess = call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    local_jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_abs_jitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    return local_jitter, local_abs_jitter, rap_jitter, ppq5_jitter, ddp_jitter

def get_shimmer_numpy(file, sr = 16000,  pitch_floor = 70, pitch_ceiling = 400):
    snd = parselmouth.Sound(values = file, sampling_frequency = sr)
    pointProcess = call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    local_shimmer =  call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    local_shimmer_db = call([snd, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([snd, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call([snd, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer =  call([snd, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([snd, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return local_shimmer, local_shimmer_db, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer

def get_hnr_numpy(file, sr = 16000, pitch_floor = 70, pitch_ceiling = 400):
    snd = parselmouth.Sound(values = file, sampling_frequency = sr)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    return hnr, harmonicity