# RIS Channel Estimation Demo

A Streamlit application for demonstrating RIS (Reconfigurable Intelligent Surface) channel estimation using deep learning models.

## Features

- Interactive channel matrix visualization
- Multiple model support (DNN, CNN, Autoencoder)
- Real-time NMSE calculation
- Heatmap visualization of channel matrices

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run RIS_Demo_Streamlit.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically

### 2. Heroku

1. Create a `Procfile`:
```
web: streamlit run RIS_Demo_Streamlit.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy using Heroku CLI or GitHub integration

### 3. Railway

1. Connect your GitHub repository to Railway
2. Railway will automatically detect and deploy your Streamlit app

### 4. Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "RIS_Demo_Streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t ris-demo .
docker run -p 8501:8501 ris-demo
```

## Project Structure

- `RIS_Demo_Streamlit.py` - Main Streamlit application
- `RIS_Advanced_Extensions.ipynb` - Advanced RIS extensions notebook
- `RIS_DNN_CNN_AE_Models.ipynb` - Deep learning models notebook
- `RIS_Channel_Simulation.ipynb` - Channel simulation notebook
- `ris_multiuser_dataset.npz` - Multi-user dataset
- `ris_channel_dataset.npz` - Channel dataset 