Repository Structure

cifar-10-production/
├── src/
│   ├── __init__.py
│   ├── model.py                # model factory (resnet50)
│   ├── train.py                # training script (train/val/early stopping/report)
│   ├── inference.py            # load_model(), preprocess(), predict_topk()
│   └── utils.py                # helpers (save/load, metrics)
│
├── api/
│   ├── __init__.py
│   ├── model_serving.py        # reuse inference functions for serving
│   └── main.py                 # FastAPI app (startup loads model)
│
├── web/
│   ├── __init__.py
│   ├── flask_app.py            # minimalist Flask wrapper / static serve
│   └── static/
│       ├── index.html
│       └── app.js
│
├── tests/
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_api.py
│
├── models/                     # trained weights (best_model.pth)
├── assets/                     # confusion matrix etc.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
