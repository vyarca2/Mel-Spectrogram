import soundfile as sf
import librosa
import numpy as np

def load_audio(file_path):
    audio, sample_rate = sf.read(file_path, dtype='float32')
    return audio

audio_file_path = "/content/audio1.wav"
audio_data = load_audio(audio_file_path)

if len(audio_data) > 0:
    sample_rate = 16000
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    f_min = 0
    f_max = sample_rate

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = mel_spectrogram.T

    output_file_path = "mel_spectrogram1.txt"
    with open(output_file_path, 'w') as output_file:
        output_file.write(str(mel_spectrogram.shape))

    print("Mel spectrogram saved to", output_file_path)

else:
    print("Failed to load audio file:", audio_file_path)
