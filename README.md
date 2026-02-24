# Face Recognition System with Anti-Spoofing

A complete face recognition system that detects faces, extracts embeddings, and verifies identity with anti-spoofing protection.  
Built with a C++ backend and a simple web frontend.

## 🚀 Quick Start (choose one)

### Option 1 – Run with Docker Compose (pull pre‑built images)

1. Create a project directory:

```bash
mkdir -p face-recognition/data
cd face-recognition
```

2. Inside the `face-recognition` directory, create a `docker-compose.yml` file with the following content:

```yaml
services:
  backend:
    image: hatfan/face-recognition-cpp:backend-latest
    container_name: face_backend
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data   # optional: persist face database
    networks:
      - face-net

  frontend:
    image: hatfan/face-recognition-cpp:frontend-latest
    container_name: face_frontend
    ports:
      - "80:80"
    networks:
      - face-net

networks:
  face-net:
```

After creating the file, your project directory should look like this:

```
face-recognition/
├── data/               # (empty folder for persistent data)
└── docker-compose.yml
```

3. Run the containers:

```bash
docker-compose up -d
```

4. Open `http://localhost` in your browser.

> No need to clone the repository – just save the compose file and run.

### Option 2 – Build locally (development)

Clone the repository and use the development compose file:

```bash
git clone https://github.com/hatfan/face-recognition-cpp.git
cd face-recognition-cpp
docker-compose up
```

The first build may take 20–30 minutes because OpenCV is compiled from source.  
After startup, access the app at `http://localhost`.

## 🧠 How It Works

1. **Frontend** captures a photo from your webcam and sends it to the backend.
2. **Backend** performs:
   - Face detection (Haar Cascade)
   - Anti‑spoofing check (MobileNet / DepthAnything)
   - Face embedding extraction (ArcFace ONNX)
   - Similarity search (FAISS)
3. Results are displayed in the web UI – you can register new faces or verify known ones.

## 📦 Tech Stack

- **Backend**: C++17, OpenCV 4.8.0, ONNX Runtime, FAISS, Boost, cpprestsdk
- **Frontend**: HTML, CSS, JavaScript (vanilla), Nginx
- **Container**: Docker, Docker Compose

## 📁 Models Used

The following pre‑trained models are included in the Docker images:

| Model | Source | Purpose |
|-------|--------|---------|
| `haarcascade_frontalface_default.xml` | OpenCV | Face detection |
| `arcfaceresnet100-8.onnx` | ONNX Model Zoo | Face embedding (512‑dim) |
| `depth_anything_v2_vits_238.onnx` | DepthAnything | depth‑based spoof detection |
| (opsional) `mobilenetv2_spoof.onnx` | Custom trained | Anti‑spoofing classification |

Model files are automatically downloaded during image build.

## ⚙️ Configuration

Thresholds and model paths can be adjusted by mounting a `config.txt` file into the container at `/app/config.txt`.  
A default config is provided inside the image. Example:

```
detector_path = /app/models/haarcascade_frontalface_default.xml
embedder_path = /app/models/arcfaceresnet100-8.onnx
spoof_model_path = /app/models/mobilenetv2_spoof.onnx
face_threshold = 0.2
spoof_threshold = 0.5
```

## 📝 Notes

- The database of registered faces is stored in `/app/data/face_db.bin`. Mount a volume if you want to keep it between container restarts.
- The first run with Option 2 (local build) takes time due to OpenCV compilation.
- For production, it is recommended to use Option 1 (pre‑built images) for faster startup and smaller image size.

## 📚 More Information

For advanced usage, API documentation, and development setup, please refer to the [GitHub repository](https://github.com/hatfan/face-recognition-cpp).
