import os
import re
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz, solve_toeplitz

def segmental_snr(clean, enhanced, frame_len=256):
    eps = 1e-10
    clean_frames = librosa.util.frame(clean, frame_length=frame_len, hop_length=frame_len)
    enhanced_frames = librosa.util.frame(enhanced, frame_length=frame_len, hop_length=frame_len)

    ssnr_list = []
    for i in range(clean_frames.shape[1]):
        c = clean_frames[:, i]
        e = enhanced_frames[:, i]
        noise = c - e
        num = np.sum(c ** 2)
        den = np.sum(noise ** 2) + eps
        if num > 0:
            ssnr_list.append(10 * np.log10(num / den + eps))
    return np.mean(ssnr_list) if len(ssnr_list) > 0 else 0.0

def llr(clean, enhanced, order=10):
    def lpc_coeffs(x, order):
        r = np.correlate(x, x, mode='full')[len(x)-1:len(x)+order]
        return solve_toeplitz((r[:-1], r[:-1]), r[1:])

    frame_len, hop = 512, 256
    clean_frames = librosa.util.frame(clean, frame_length=frame_len, hop_length=hop)
    enhanced_frames = librosa.util.frame(enhanced, frame_length=frame_len, hop_length=hop)

    llrs = []
    for i in range(clean_frames.shape[1]):
        c = clean_frames[:, i]
        e = enhanced_frames[:, i]
        try:
            a_clean = lpc_coeffs(c, order)
            a_enh = lpc_coeffs(e, order)
            num = np.dot(a_clean - a_enh, a_clean - a_enh)
            den = np.dot(a_clean, a_clean)
            llrs.append(num / (den + 1e-10))
        except:
            continue
    return np.mean(llrs) if len(llrs) > 0 else 0.0

def wss(clean, enhanced):
    n_fft, hop = 512, 256
    clean_mag = np.abs(librosa.stft(clean, n_fft=n_fft, hop_length=hop))
    enhanced_mag = np.abs(librosa.stft(enhanced, n_fft=n_fft, hop_length=hop))
    min_len = min(clean_mag.shape[1], enhanced_mag.shape[1])
    clean_mag, enhanced_mag = clean_mag[:, :min_len], enhanced_mag[:, :min_len]
    clean_slope = np.diff(np.log(clean_mag + 1e-10), axis=0)
    enhanced_slope = np.diff(np.log(enhanced_mag + 1e-10), axis=0)
    return np.mean((clean_slope - enhanced_slope) ** 2)

def compute_metrics(clean_dir, enhanced_dir, target_sr=16000, min_len_threshold=1000):
    pesq_scores, stoi_scores, ssnr_scores, llr_scores, wss_scores = [], [], [], [], []

    # Build dictionary of enhanced files keyed by full ID (e.g., p232_001)
    enh_files_dict = {}
    for fname in os.listdir(enhanced_dir):
        match = re.search(r'p\d+_\d+', fname)  # full ID
        if match:
            file_id = match.group()
            enh_files_dict[file_id] = fname

    # Iterate over clean files
    for i, clean_fname in enumerate(os.listdir(clean_dir)):
        match = re.search(r'p\d+_\d+', clean_fname)  # full ID
        if not match:
            print(f"Skipping {clean_fname}: no valid ID found")
            continue
        file_id = match.group()

        if file_id not in enh_files_dict:
            print(f"Skipping {clean_fname}: no matching enhanced file for ID {file_id}")
            continue

        enh_fname = enh_files_dict[file_id]
        clean_path = os.path.join(clean_dir, clean_fname)
        enh_path = os.path.join(enhanced_dir, enh_fname)

        # Load audio at original SR
        clean, sr_clean = librosa.load(clean_path, sr=None)
        enhanced, sr_enh = librosa.load(enh_path, sr=None)

        print(f"[{i}] Processing ID {file_id}: Clean={clean_fname} ({len(clean)} samples), Enhanced={enh_fname} ({len(enhanced)} samples)")

        if len(clean) < min_len_threshold or len(enhanced) < min_len_threshold:
            print(f"Skipping ID {file_id}: too short")
            continue

        # Resample to target SR if needed
        if sr_clean != target_sr:
            clean = librosa.resample(clean, orig_sr=sr_clean, target_sr=target_sr)
        if sr_enh != target_sr:
            enhanced = librosa.resample(enhanced, orig_sr=sr_enh, target_sr=target_sr)

        # Truncate to same length
        min_len = min(len(clean), len(enhanced))
        clean, enhanced = clean[:min_len], enhanced[:min_len]

        # PESQ
        try:
            pesq_scores.append(pesq(target_sr, clean, enhanced, 'wb'))
        except:
            pesq_scores.append(np.nan)

        # STOI
        try:
            stoi_scores.append(stoi(clean, enhanced, target_sr, extended=False))
        except:
            stoi_scores.append(np.nan)

        # SSNR
        ssnr_scores.append(segmental_snr(clean, enhanced))

        # LLR
        llr_scores.append(llr(clean, enhanced))

        # WSS
        wss_scores.append(wss(clean, enhanced))

    # Safe average
    def safe_mean(arr):
        arr = [a for a in arr if not np.isnan(a)]
        return np.mean(arr) if arr else np.nan

    pesq_avg = safe_mean(pesq_scores)
    stoi_avg = safe_mean(stoi_scores)
    ssnr_avg = safe_mean(ssnr_scores)
    llr_avg = safe_mean(llr_scores)
    wss_avg = safe_mean(wss_scores)

    # SEGAN composite metrics
    csig = 3.093 - 1.029 * llr_avg + 0.603 * pesq_avg + 0.009 * wss_avg
    cbak = 1.634 + 0.478 * pesq_avg - 0.007 * wss_avg + 0.063 * ssnr_avg
    covl = 1.594 + 0.805 * pesq_avg - 0.512 * llr_avg + 0.007 * ssnr_avg

    return {
        "PESQ": pesq_avg,
        "STOI": stoi_avg,
        "SSNR": ssnr_avg,
        "LLR": llr_avg,
        "WSS": wss_avg,
        "CSIG": csig,
        "CBAK": cbak,
        "COVL": covl,
    }

if __name__ == "__main__":
    clean_dir = "../test_clean/"
    enhanced_dir = "../results/"
    metrics = compute_metrics(clean_dir, enhanced_dir, target_sr=16000)
    print(metrics)
