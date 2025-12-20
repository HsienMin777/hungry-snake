FROM python:3.9-slim

# 安裝系統依賴 (Pygame 執行所需)
# 修正：將 libgl1-mesa-glx 替換為 libgl1
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. 複製依賴檔案並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 複製 App 資料夾內所有內容到容器的 /app
COPY App/ . 

# 執行程式
CMD ["python", "app.py"]