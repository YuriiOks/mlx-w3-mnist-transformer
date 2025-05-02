# MNIST Digit Classifier (Transformer) - Streamlit Utilities
# File: app/utils_streamlit.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Streamlit-specific utility functions (plotting, caching, etc.).
# Created: 2025-05-02
# Updated: 2025-05-02

import streamlit as st
import matplotlib.pyplot as plt

def plot_probabilities(
    probs,
    phase,
    result_area
):
    """
    Displays a probability bar chart in the Streamlit app.

    Args:
        probs (array-like): Probabilities to plot. For phase 1, a 1D array of 
            length 10. For phase 2, a 2D array (4, 10) representing probabilities 
            for each position.
        phase (int): Indicates the phase of the app. If 1, plots a single 
            probability bar chart. If 2, plots four bar charts (one per position).
        result_area (streamlit.delta_generator.DeltaGenerator): Streamlit 
            container to display the plot in.

    Returns:
        None. The function displays the plot in the Streamlit app.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    with result_area:
        if phase == 1:
            labels = [str(i) for i in range(10)]
            fig, ax = plt.subplots(figsize=(6, 2))
            bars = ax.bar(labels, probs.flatten(), color='skyblue')
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Output Probabilities")
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.02,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
            st.pyplot(fig)
        elif phase == 2:
            labels = [str(i) for i in range(10)]
            fig, axes = plt.subplots(2, 2, figsize=(7, 4))
            fig.suptitle("Output Probabilities per Position")
            pos = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
            probs_flat = np.array(probs).reshape(4, 10)
            for i, ax in enumerate(axes.flat):
                bars = ax.bar(labels, probs_flat[i], color='skyblue')
                ax.set_title(pos[i])
                ax.set_ylim(0, 1)
                ax.set_xticks(range(10))
                ax.tick_params(axis='x', labelsize=8)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height + 0.02,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=6
                    )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig)
        else:
            st.info("Probability plot not implemented for this phase.")

# Add more Streamlit-specific helpers as needed
