import os
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi

def segmental_snr(clean, enhanced, frame_len=256):
    """Compute Segmental SNR (SSNR)."""
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


def compute_metrics(clean_dir, enhanced_dir, sr=48000):
    pesq_scores, stoi_scores, ssnr_scores = [], [], []

    for fname in os.listdir(clean_dir):
        clean_path = os.path.join(clean_dir, fname)

        # Map clean filename to enhanced filename
        base = os.path.splitext(fname)[0]   # e.g. p232_001
        enh_fname = base + "_fine_tuned.wav"
        enh_path = os.path.join(enhanced_dir, enh_fname)

        if not os.path.exists(enh_path):
            print(f"⚠️ Skipping {fname}, no enhanced match found")
            continue

        # Load audio at full resolution (48kHz)
        clean, _ = librosa.load(clean_path, sr=sr)
        enhanced, _ = librosa.load(enh_path, sr=sr)

        # Truncate to same length
        min_len = min(len(clean), len(enhanced))
        clean, enhanced = clean[:min_len], enhanced[:min_len]

        # === PESQ requires 16kHz ===
        target_sr = 16000
        clean_16k = librosa.resample(clean, orig_sr=sr, target_sr=target_sr)
        enhanced_16k = librosa.resample(enhanced, orig_sr=sr, target_sr=target_sr)

        try:
            pesq_score = pesq(target_sr, clean_16k, enhanced_16k, 'wb')
            pesq_scores.append(pesq_score)
        except Exception as e:
            print(f"⚠️ Skipping PESQ for {fname} due to error: {e}")

        # === STOI at 48kHz ===
        stoi_score = stoi(clean, enhanced, sr, extended=False)
        stoi_scores.append(stoi_score)

        # === SSNR at 48kHz ===
        ssnr_score = segmental_snr(clean, enhanced)
        ssnr_scores.append(ssnr_score)

    # Average metrics
    pesq_avg = np.mean(pesq_scores) if pesq_scores else 0.0
    stoi_avg = np.mean(stoi_scores) if stoi_scores else 0.0
    ssnr_avg = np.mean(ssnr_scores) if ssnr_scores else 0.0

    # SEGAN composite metrics
    csig = 3.093 - 1.029 * pesq_avg + 0.603 * ssnr_avg
    cbak = 1.634 + 0.478 * pesq_avg + 0.007 * ssnr_avg
    covl = 1.594 + 0.805 * pesq_avg + 0.512 * ssnr_avg

    return {
        "PESQ": pesq_avg,
        "STOI": stoi_avg,
        "SSNR": ssnr_avg,
        "CSIG": csig,
        "CBAK": cbak,
        "COVL": covl,
    }


# Example usage
if __name__ == "__main__":
    clean_dir = "../test_clean/"
    enhanced_dir = "../results/"
    metrics = compute_metrics(clean_dir, enhanced_dir, sr=48000)
    print(metrics)
