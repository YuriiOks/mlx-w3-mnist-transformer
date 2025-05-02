# MNIST Digit Classifier (Transformer) - Sidebar Controls
# File: app/sidebar.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Handles sidebar controls for framework, phase, and model selection.
# Created: 2025-05-02
# Updated: 2025-05-02

import streamlit as st
import os
import re
from pathlib import Path

def parse_run_name(run_name):
    """Extracts Phase, Epochs, LR, Batch Size from standard run names."""
    match = re.match(r"(PT|MLX)_Phase(\d+)_E(\d+)_LR([\d.eE+-]+)_B(\d+).*", run_name)
    if match:
        return {
            'framework': match.group(1),
            'phase': match.group(2),
            'epochs': match.group(3),
            'lr': match.group(4),
            'batch': match.group(5),
            'full_name': run_name
        }
    return {'full_name': run_name}

def sidebar_controls(config):
    """Handles sidebar controls: framework, phase, model selection, load button."""
    st.sidebar.header("✨ Chef's Station ✨")
    st.sidebar.markdown("---")
    # 1. Framework Selection
    st.sidebar.subheader("1. Select Framework 🤖/🍎")
    framework_options = []
    try: import torch; framework_options.append("🤖 PyTorch")
    except ImportError: pass
    try: import mlx.core; framework_options.append("🍎 MLX")
    except ImportError: pass
    if not framework_options: st.sidebar.error("No ML framework found!"); st.stop()
    selected_framework = st.sidebar.selectbox(
        "Choose ML Framework:",
        framework_options,
        key="framework_select"
    )
    st.sidebar.markdown("---")
    # 2. Phase Selection
    st.sidebar.subheader("2. Select Cooking Phase 🍲")
    phase_options = {
        1: "🧁 Phase 1: Single Digit",
        2: "🍽️ Phase 2: 2x2 Grid",
        3: "🍜 Phase 3: Sequence"
    }
    selected_phase = st.sidebar.selectbox(
        "Choose Phase:",
        options=list(phase_options.keys()),
        format_func=lambda x: phase_options.get(x, f"Phase {x}"),
        key="phase_select"
    )
    st.sidebar.markdown("---")
    # 3. Run Selection (with parsed details)
    st.sidebar.subheader("3. Select Trained Model 💾")
    model_base_dir = config.get('paths', {}).get('model_save_dir', 'models/mnist_vit')
    prefix = "PT_" if selected_framework == "PyTorch" else "MLX_"
    phase_prefix = f"{prefix}Phase{selected_phase}_"
    available_runs = []
    run_display_options = {}
    try:
        if Path(model_base_dir).exists():
            all_dirs = sorted([
                d.name for d in Path(model_base_dir).iterdir()
                if d.is_dir() and d.name.startswith(phase_prefix)
            ], reverse=True)
            for run_name in all_dirs:
                 details = parse_run_name(run_name)
                 parts = []
                 if 'epochs' in details: parts.append(f"E{details['epochs']}")
                 if 'lr' in details: parts.append(f"LR {details['lr']}")
                 if 'batch' in details: parts.append(f"B{details['batch']}")
                 suffix = f"...{run_name[-8:]}" if len(run_name) > 25 else run_name
                 parts.append(f"({suffix})")
                 display_name = " | ".join(parts)
                 if display_name not in run_display_options:
                      run_display_options[display_name] = run_name
                 else:
                      display_name += f"_{run_name[-12:]}"
                      run_display_options[display_name] = run_name
            available_runs = list(run_display_options.keys())
    except Exception as e:
        available_runs = []
    selected_display_name = None
    model_run_name = None
    if not available_runs:
        st.sidebar.info(f"No trained {selected_framework} Phase {selected_phase} models found.")
    else:
        selected_display_name = st.sidebar.selectbox(
            f"Select {selected_framework[1:]} Phase {selected_phase} Run:",
            options=available_runs
        )
        if selected_display_name:
            model_run_name = run_display_options[selected_display_name]
    st.sidebar.markdown("---")
    # 4. Load Model Button
    st.sidebar.subheader("4. Load Model ▶️")
    load_model_clicked = st.sidebar.button("Load Selected Model", key="load_button", disabled=(model_run_name is None))
    return {
        'framework': selected_framework,
        'phase': selected_phase,
        'model_run_name': model_run_name,
        'load_model_clicked': load_model_clicked,
        'model_base_dir': model_base_dir
    }
