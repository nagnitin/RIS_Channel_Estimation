import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# -------------------------------
# Correct Dummy Model (outputs 64)
# -------------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 32),  # Hidden layer
            nn.ReLU(),
            nn.Linear(32, 64)  # ✅ Output must match H_true (64)
        )

    def forward(self, x):
        return self.fc(x)

# Register dummy models
models = {
    "DNN": DummyModel(),
    "CNN": DummyModel(),
    "Autoencoder": DummyModel()
}

# -------------------------------
# Load Simulated Data (64-length H_true)
# -------------------------------
@st.cache_data
def load_sample_data():
    H_true = np.random.randn(64)  # True channel matrix (8x8) flattened
    y_real, y_imag = np.random.randn(1), np.random.randn(1)
    X = torch.tensor([y_real[0], y_imag[0]], dtype=torch.float32).unsqueeze(0)
    return H_true, X

# -------------------------------
# Plot Heatmaps
# -------------------------------
def plot_heatmaps(true_H, pred_H):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(np.abs(true_H.reshape(8, 8)), ax=axes[0], cmap='viridis')
    axes[0].set_title("True |H|")

    sns.heatmap(np.abs(pred_H.reshape(8, 8)), ax=axes[1], cmap='viridis')
    axes[1].set_title("Predicted |H|")

    st.pyplot(fig)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="RIS Channel Estimation", layout="centered")
st.title("📡 RIS-Assisted Channel Estimation Demo")
st.markdown("Estimate RIS channel matrix using Deep Learning models")

model_name = st.selectbox("Select Model", list(models.keys()))
H_true, X = load_sample_data()

if st.button("Run Estimation"):
    model = models[model_name]
    model.eval()

    with torch.no_grad():
        H_pred = model(X).squeeze().numpy()

    # ✅ Confirm shapes match
    if H_pred.shape != H_true.shape:
        st.error(f"Shape mismatch: H_true={H_true.shape}, H_pred={H_pred.shape}")
    else:
        nmse = np.mean((H_true - H_pred)**2) / (np.mean(H_true**2) + 1e-10)

        st.metric(label="📈 NMSE", value=f"{nmse:.4f}")
        st.subheader("🔍 Channel Matrix Heatmaps")
        plot_heatmaps(H_true, H_pred)

st.caption("Demo using DummyModel. Replace with trained PyTorch models for real evaluation.")
