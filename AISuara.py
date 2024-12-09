import pyaudio
import torch
import torchaudio
import torchaudio.transforms as T
import struct
import numpy as np
import socket

# Konfigurasi PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Inisialisasi PyAudio
p = pyaudio.PyAudio()

# Buka stream dari perangkat audio virtual (misalnya "Cable Output")
# Gantilah perangkat dengan perangkat audio virtual yang digunakan untuk Discord
input_device_index = None
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if "CABLE Output" in device_info["name"]:  # Nama perangkat audio virtual Anda
        input_device_index = i
        break

if input_device_index is None:
    raise ValueError("Perangkat audio virtual tidak ditemukan. Pastikan VB-Cable atau perangkat serupa sudah terpasang.")

# Buka stream mic dari perangkat audio virtual (Cable Output)
stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                   input=True, input_device_index=input_device_index,
                   frames_per_buffer=CHUNK)

def ubah_suara(data, jenis_suara):
    # Pilih efek berdasarkan jenis suara
    if jenis_suara == "pria":
        efek = T.PitchShift(n_steps=-2, sample_rate=RATE)
    elif jenis_suara == "wanita":
        efek = T.PitchShift(n_steps=2, sample_rate=RATE)
    elif jenis_suara == "robot":
        efek = T.Resample(orig_freq=RATE, new_freq=int(RATE * 1.5))
    else:
        raise ValueError("Jenis suara tidak didukung")

    # Mengonversi data menjadi tensor float32
    tensor_suara = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # Format tensor [1, 1, N]
    suara_ubah = efek(tensor_suara)  # Terapkan efek suara
    return suara_ubah

# Konfigurasi Socket (UDP)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 10000)  # Server address yang akan menerima data

# Main loop
jenis_suara = "wanita"
while True:
    # Baca data dari perangkat audio virtual (misalnya "Cable Output")
    data = stream_in.read(CHUNK)
    
    # Mengonversi byte menjadi integer (16-bit)
    data_int = np.array(struct.unpack('h' * CHUNK, data), dtype=np.float32)
    
    # Normalisasi data ke rentang (-1, 1)
    data_int /= 32768.0  # Rentang normalisasi (-1, 1)
    
    # Terapkan efek perubahan suara
    suara_ubah = ubah_suara(data_int, jenis_suara)
    suara_ubah = suara_ubah.detach().cpu().flatten().numpy()

    # Normalisasi suara_ubah ke rentang (-1, 1) untuk menghindari distorsi
    max_val = np.max(np.abs(suara_ubah))  # Cari nilai maksimum dari suara_ubah
    if max_val > 0:
        suara_ubah /= max_val  # Menjaga agar tetap dalam rentang aman

    # Terapkan smoothing (filter) untuk mengurangi noise pada suara
    suara_ubah = np.convolve(suara_ubah, np.ones(5)/5, mode='same')  # Gunakan rata-rata untuk smoothing

    # Kirim suara yang diproses ke server via socket UDP
    try:
        sock.sendto(suara_ubah.tobytes(), server_address)
    except Exception as e:
        print(f"Terjadi kesalahan saat mengirim data: {e}")

# Tutup stream dan PyAudio setelah loop selesai (tidak tercapai dalam kode ini karena loop berjalan terus)
stream_in.stop_stream()
stream_in.close()
p.terminate()

# Menutup socket
sock.close()
