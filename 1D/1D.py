# Usual Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# Librosa (the mother of audio files)
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

import os

# Đường dẫn tới thư mục chứa thư mục con chứa file âm thanh
general_path = 'D:\\NLN\\audioData\\data\\wav\\test_train_random10s'

# Duyệt qua tất cả các thư mục con
for subdir, _, files in os.walk(general_path):
    for file in files:
        if file.endswith('.wav'):
            # Lấy tên thư mục chứa file
            parent_folder_name = os.path.basename(subdir)

            file_path = os.path.join(subdir, file)  # Đường dẫn đầy đủ tới tệp âm thanh

            # Importing 1 file
            y, sr = librosa.load(file_path)

            # Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
            audio_file, _ = librosa.effects.trim(y)

            # Tạo đường dẫn cho thư mục đầu ra dựa trên tên thư mục chứa file
            output_dir = os.path.join('D:\\NLN\\audioData\\data\\test_train_random10s', parent_folder_name)
            os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục đầu ra nếu chưa tồn tại

            # Tạo tên file cho tệp văn bản dựa trên tên tệp âm thanh
            output_text_path = os.path.join(output_dir, f'{os.path.splitext(file)[0]}.txt')

            # Lưu mảng 1D (dữ liệu âm thanh) thành tệp văn bản
            np.savetxt(output_text_path, audio_file, fmt='%f', delimiter='\n')
