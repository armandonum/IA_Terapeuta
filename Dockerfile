# Usar una imagen base de Python con CUDA para GPU (opcional, cambiar a python:3.10-slim si no usas GPU)
FROM python:3.10-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/La_Paz

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    libportaudio2 \
    libportaudiocpp0 \
    espeak-ng \
    libespeak-ng1 \
    alsa-utils \
    pulseaudio \
    libasound2-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar Coqui TTS sin dependencias (como indica tu comentario)
RUN pip install --no-deps TTS==0.22.0

# Copiar el resto de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p session_videos models static templates emotion_processor examples output

# Exponer el puerto de Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app_integrated.py"]