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

2. Run the app (use a free port, e.g. 8503):
```bash
streamlit run RIS_Demo_Streamlit.py --server.port 8503
```

## Deployment (Recommended: Streamlit Cloud)

This project is designed for free deployment on [Streamlit Cloud](https://share.streamlit.io):

1. Push your code to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository: `nagnitin/RIS_Channel_Estimation`
5. Set the main file path: `RIS_Demo_Streamlit.py`
6. Click "Deploy"

Your app will be live on a public URL provided by Streamlit Cloud!

## Project Structure

- `RIS_Demo_Streamlit.py` - Main Streamlit application
- `RIS_Advanced_Extensions.ipynb` - Advanced RIS extensions notebook
- `RIS_DNN_CNN_AE_Models.ipynb` - Deep learning models notebook
- `RIS_Channel_Simulation.ipynb` - Channel simulation notebook
- `ris_multiuser_dataset.npz` - Multi-user dataset
- `ris_channel_dataset.npz` - Channel dataset
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `.github/workflows/deploy.yml` - GitHub Actions workflow for automated testing

