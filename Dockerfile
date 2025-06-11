# ---- 1. Base image ---------------------------------------------------------
FROM python:3.12-slim

# ---- 2. Hệ thống gói cần thiết --------------------------------------------
RUN apt-get update && \
    # ffmpeg và sox cần cho torchaudio & chuyển đổi MP3/WAV
    apt-get install -y --no-install-recommends ffmpeg sox && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- 3. Thiết lập thư mục làm việc ----------------------------------------
WORKDIR /app

# ---- 4. Cài đặt phụ thuộc Python -----------------------------------------
# Sao chép trước requirements để cache layer pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 5. Copy mã nguồn ------------------------------------------------------
COPY . .

# ---- 6. Chạy bot -----------------------------------------------------------
# PYTHONUNBUFFERED để log realtime
ENV PYTHONUNBUFFERED=1
CMD ["python", "bot_song_id.py"]
