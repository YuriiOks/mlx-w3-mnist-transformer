# MNIST Digit Classifier (Transformer) - Streamlit Utilities
# File: app/utils_streamlit.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Streamlit-specific utility functions (plotting, caching, etc.).
# Created: 2025-05-02
# Updated: 2025-05-02

# Streamlit-specific utility functions for the MNIST ViT app
import streamlit as st
import matplotlib.pyplot as plt

def plot_probabilities(probs, phase, result_area):
    """Displays probability bar chart in the Streamlit app."""
    with result_area:
        st.write("Probability plot placeholder.")
        # You can implement phase-specific plotting here
        # Example: plt.bar(...), st.pyplot(...)

# Add more Streamlit-specific helpers as needed
