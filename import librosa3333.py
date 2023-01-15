import librosa
import matplotlib.pyplot as plt

# Video dosyasını yükle
filename = 'video.mp4'
y, sr = librosa.load(filename)

# Ses düzeylerini analiz et
melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Konuşma olmadığı yerleri tespit et
silent_frames = librosa.util.find_silent_frames(melspec,threshold=30)

# Sessiz kareleri görüntüle
plt.figure(figsize=(12,4))
librosa.display.specshow(librosa.power_to_db(melspec,ref=np.max),x_axis='time',sr=sr,hop_length=512)
plt.vlines(silent_frames/sr,0,sr, color='r', linestyle='dotted')
plt.show()