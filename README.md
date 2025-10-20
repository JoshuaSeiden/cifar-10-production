# 🧠 CIFAR-10 Image Classifier

This project demonstrates a complete end-to-end machine learning deployment pipeline, from training to inference and hosting.
It features a fine-tuned ResNet-18 image classifier trained on the CIFAR-10 dataset and served via a FastAPI backend with a lightweight HTML/JavaScript frontend.

🌐 Live Demo: https://jseiden-cifar-10.onrender.com/

# 🚀 Features
- 🔍 Model Training: Fine-tuned ResNet-18 on the CIFAR-10 dataset using PyTorch and CUDA acceleration.
- 🧩 Model Serving: FastAPI-based REST service with /predict, /classes, and /health endpoints.
- 🌐 Frontend: Minimal web UI (HTML/CSS/JS) for uploading images and visualizing predictions.
- 🐋 Containerized: Fully reproducible environment using Docker.
- 🔄 CI/CD: Automated testing and deployment pipeline via GitHub Actions → Render.
- ☁️ Model Hosting: Pre-trained weights stored and versioned on Hugging Face Hub.
- 🧪 Testing: Unit tests for API and inference logic with PyTest.

# 🧰 Tech Stack
| Category           | Tools / Libraries      |
| ------------------ | ---------------------- |
| **Model Training** | PyTorch, Torchvision   |
| **Serving**        | FastAPI, Uvicorn       |
| **Frontend**       | HTML, CSS, JavaScript  |
| **Deployment**     | Docker, Render         |
| **Automation**     | GitHub Actions, PyTest |
| **Model Hosting**  | Hugging Face Hub       |

# ⚙️ API Endpoints
| Endpoint   | Method | Description                                                    |
| ---------- | ------ | -------------------------------------------------------------- |
| `/classes` | `GET`  | Returns the CIFAR-10 image classes (populates UI explanation). |
| `/health`  | `GET`  | Returns service status (used for health checks).               |
| `/predict` | `POST` | Accepts an image file and returns the top-3 predicted classes. |

# Example
curl -X POST "https://jseiden-cifar-10.onrender.com/predict" \
  -F "file=@sample_image.jpg"

# Response
{
  "predictions": [
    {"class": "cat", "prob": 0.92},
    {"class": "dog", "prob": 0.05},
    {"class": "airplane", "prob": 0.02}
  ]
}

# 🔄 CI/CD Overview
- ✅ CI: Every push to main runs automated tests with PyTest.
- 🚀 CD: On successful test completion, the pipeline automatically deploys the latest version to Render using the containerized image.

# 🧠 Training Overview
- Dataset: CIFAR-10
- Model: ResNet-18
- Fine-tuned on 10 classes
- Optimizer: Adam
- Loss: Cross-Entropy
- Framework: PyTorch (CUDA enabled)

# 🩺 Health Check
Check service uptime:
curl https://jseiden-cifar-10.onrender.com/health

Response:
{"status": "ok"}

# 🧑‍💻 Author
Joshua Seiden
Machine Learning Engineer • Data Science Enthusiast