"""
Ocular Disease Detection System - Blue/Teal Theme
Multi-page app: Prediction + Evaluation
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from src.ensemble import EnsemblePredictor
import config
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import json

st.set_page_config(
    page_title="Ocular Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
#  RED THEME CSS - ENHANCED (Original)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { 
        background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 50%, #1a0505 100%); 
        animation: bgShift 15s ease infinite;
    }
    .stApp { 
        background: linear-gradient(135deg, #1a0505 0%, #2d0a0a 50%, #1a0505 100%); 
    }
    
    @keyframes bgShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .title-box {
        background: linear-gradient(135deg, #7f1d1d, #991b1b, #dc2626);
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 30px;
        border: 3px solid #ef4444;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.4);
        animation: pulse 3s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 10px 40px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 15px 60px rgba(239, 68, 68, 0.6); }
    }
    .title-box h1 {
        color: #fca5a5;
        font-size: 3em;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 0 30px rgba(252, 165, 165, 0.8);
        letter-spacing: 1px;
    }
    .title-box p {
        color: #fecaca;
        font-size: 1.3em;
        margin: 15px 0 0 0;
        font-weight: 400;
    }

    .metric-card {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        border: 2px solid #dc2626;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.5);
    }
    .metric-card h3 { 
        color: #ef4444; 
        margin: 0; 
        font-size: 2.5em;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.6);
    }
    .metric-card p { 
        color: #fca5a5; 
        margin: 8px 0 0 0; 
        font-size: 1em;
        font-weight: 600;
    }

    .result-box {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-radius: 25px;
        padding: 30px;
        border: 3px solid #ef4444;
        text-align: center;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.4);
    }
    .result-box h2 { 
        color: #fca5a5; 
        margin: 0;
        font-size: 2.3em;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(252, 165, 165, 0.7);
    }
    .result-box h3 { 
        color: #fecaca; 
        margin: 15px 0; 
        font-size: 1.7em;
        font-weight: 600;
    }

    .disease-card {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-radius: 15px;
        padding: 22px;
        margin: 12px 0;
        border-left: 6px solid #ef4444;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.3);
        transition: all 0.3s ease;
    }
    .disease-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.5);
    }
    .disease-card h4 { 
        color: #fca5a5; 
        margin: 0 0 10px 0;
        font-size: 1.2em;
        font-weight: 600;
    }
    .disease-card p { 
        color: #fecaca; 
        margin: 0; 
        font-size: 1em; 
        line-height: 1.6;
    }

    .stProgress > div > div { 
        background: linear-gradient(90deg, #dc2626, #ef4444, #f87171) !important; 
    }
    
    .sidebar .sidebar-content { 
        background: linear-gradient(180deg, #450a0a, #1a0505); 
    }
    
    div[data-testid="stSidebarNav"] {
        background: linear-gradient(180deg, #450a0a, #1a0505);
    }
    
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #450a0a, #1a0505);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border-radius: 12px;
        color: #fca5a5;
        border: 2px solid #991b1b;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #dc2626;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        border: 2px solid #ef4444;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .uploadedFile {
        border: 2px dashed #dc2626 !important;
        background: rgba(127, 29, 29, 0.2) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #fca5a5 !important;
    }
    
    .stMarkdown {
        color: #fecaca;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DISEASE INFO
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_INFO = {
    "Age-related Macular Degeneration": {
        "icon": "👴", "desc": "Deterioration of the macula causing central vision loss.",
        "symptoms": "Blurred central vision, distorted lines, dark spots",
        "treatment": "Anti-VEGF injections, laser therapy, vitamins",
        "severity": "High", "color": "#ef4444"
    },
    "Cataracts": {
        "icon": "🌫️", "desc": "Clouding of the eye lens leading to blurry vision.",
        "symptoms": "Cloudy vision, glare sensitivity, faded colors",
        "treatment": "Surgical removal and lens replacement",
        "severity": "Medium", "color": "#f59e0b"
    },
    "Diabetic Retinopathy": {
        "icon": "🩸", "desc": "Diabetes-related damage to retinal blood vessels.",
        "symptoms": "Floaters, blurred vision, dark areas, vision loss",
        "treatment": "Blood sugar control, laser treatment, vitrectomy",
        "severity": "High", "color": "#ef4444"
    },
    "Glaucoma": {
        "icon": "👁️", "desc": "Damage to optic nerve from high eye pressure.",
        "symptoms": "Gradual peripheral vision loss, tunnel vision",
        "treatment": "Eye drops, laser therapy, surgery",
        "severity": "High", "color": "#ef4444"
    },
    "Hypertension": {
        "icon": "💉", "desc": "High blood pressure damaging retinal blood vessels.",
        "symptoms": "Usually no symptoms, detected during eye exam",
        "treatment": "Blood pressure medication, lifestyle changes",
        "severity": "Medium", "color": "#f59e0b"
    },
    "Myopia": {
        "icon": "🔍", "desc": "Nearsightedness — difficulty seeing distant objects.",
        "symptoms": "Blurry distant vision, squinting, headaches",
        "treatment": "Glasses, contact lenses, LASIK surgery",
        "severity": "Low", "color": "#10b981"
    },
    "Normal": {
        "icon": "✅", "desc": "No ocular disease detected. Eyes appear healthy.",
        "symptoms": "No symptoms",
        "treatment": "Regular eye checkups recommended",
        "severity": "None", "color": "#10b981"
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    return EnsemblePredictor(
        class_indices_path=config.CLASS_INDICES_PATH,
        model_dir=config.SAVED_MODELS_DIR,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
def get_gradcam_heatmap(model, img_array, class_idx):
    """Robust Grad-CAM that works for all 5 models"""
    try:
        img_tensor = tf.cast(np.expand_dims(img_array, 0), tf.float32)

        # Find last conv layer — search all layers including nested backbones
        last_conv_output = None
        for layer in reversed(model.layers):
            # Direct conv layer in model
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.DepthwiseConv2D)):
                last_conv_output = layer.output
                break
            # Nested backbone (ResNet, DenseNet, MobileNet, VGG)
            if hasattr(layer, 'layers'):
                for sub in reversed(layer.layers):
                    if isinstance(sub, (tf.keras.layers.Conv2D,
                                       tf.keras.layers.DepthwiseConv2D)):
                        last_conv_output = sub.output
                        break
                if last_conv_output is not None:
                    break

        if last_conv_output is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    except Exception:
        # Fallback: saliency map using input gradients
        try:
            img_var = tf.Variable(tf.cast(np.expand_dims(img_array, 0), tf.float32))
            with tf.GradientTape() as tape:
                pred = model(img_var, training=False)
                loss = pred[:, class_idx]
            grads    = tape.gradient(loss, img_var)
            saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1)
            saliency = saliency / (tf.reduce_max(saliency) + 1e-8)
            return saliency.numpy()
        except Exception:
            return None

def overlay_heatmap(original_img, heatmap, alpha=0.45):
    img = np.array(original_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    hm = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hm = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, hm, alpha, 0)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👁️ Ocular AI")
    st.markdown("---")
    
    page = st.radio("Navigation", ["🔍 Prediction", "📊 Evaluation"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### Patient Info")
    patient_name = st.text_input("Name", placeholder="Enter name")
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=25)
    
    st.markdown("---")
    st.markdown("### Models Loaded")
    try:
        predictor = load_models()
        for name in predictor.loaded_model_names:
            st.markdown(f"✅ {name}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="title-box">
    <h1>👁️ Ocular Disease Detection System</h1>
    <p>AI-powered detection using 11 deep learning models</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Prediction":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>{len(predictor.loaded_model_names)}</h3>
            <p>AI Models</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <h3>7</h3><p>Disease Classes</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <h3>92%+</h3><p>Target Accuracy</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <h3>17K+</h3><p>Training Images</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload Fundus Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a retinal fundus image"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))) / 255.0

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, width=700)
            st.caption(f"Size: {image.size[0]}x{image.size[1]} px")

        with col2:
            st.subheader("🤖 AI Diagnosis")
            with st.spinner("Analyzing..."):
                try:
                    predictions, predicted_class = predictor.predict(img_array)
                    confidence = predictions[predicted_class]
                    info = DISEASE_INFO.get(predicted_class, {})

                    patient_display = f"Patient: {patient_name}" if patient_name else ""
                    age_display = f" | Age: {patient_age}" if patient_name else ""
                    patient_html = f'<p style="color:#fecaca;font-size:1em">{patient_display}{age_display}</p>' if patient_name else ""
                    time_html = f'<p style="color:#fecaca;font-size:0.9em">🕐 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
                    result_html = f"""
                    <div class="result-box">
                        <h2>{info.get('icon','')} {predicted_class}</h2>
                        <h3>Confidence: {confidence:.2%}</h3>
                        <p style="color:#fca5a5">Severity: {info.get('severity','N/A')}</p>
                        {patient_html}
                        {time_html}
                    </div>
                    """
                    st.markdown(result_html, unsafe_allow_html=True)

                    st.markdown("#### All Probabilities")
                    for disease, conf in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.progress(float(conf), text=disease)
                        with col_b:
                            st.write(f"`{conf:.2%}`")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

        st.markdown("---")
        st.subheader(f"ℹ️ About: {predicted_class}")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"""<div class="disease-card">
                <h4>📋 Description</h4>
                <p>{info.get('desc','N/A')}</p>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class="disease-card">
                <h4>⚠️ Symptoms</h4>
                <p>{info.get('symptoms','N/A')}</p>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class="disease-card">
                <h4>💊 Treatment</h4>
                <p>{info.get('treatment','N/A')}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.checkbox("🔥 Show Grad-CAM Heatmaps", help="Visualize which regions the AI focused on"):
            st.subheader("🔥 Grad-CAM — Model Attention Maps")
            st.caption("Red/yellow regions = high attention | Blue/purple = low attention")
            
            class_names = list(predictor.class_indices.keys())
            pred_idx = class_names.index(predicted_class) if predicted_class in class_names else 0

            models_to_show = [
                ("EfficientNetB3", predictor.models.get("efficientnetb3")),
                ("ResNet50",       predictor.models.get("resnet50")),
                ("DenseNet121",    predictor.models.get("densenet121")),
                ("MobileNetV2",    predictor.models.get("mobilenetv2")),
                ("VGG16",          predictor.models.get("vgg16")),
            ]
            models_to_show = [(n, m) for n, m in models_to_show if m is not None]

            # Show 3 per row
            for row_start in range(0, len(models_to_show), 3):
                row_models = models_to_show[row_start:row_start+3]
                gcols = st.columns(len(row_models))
                for i, (mname, mmodel) in enumerate(row_models):
                    with gcols[i]:
                        st.markdown(f"### {mname}")
                        with st.spinner(f"Generating {mname} heatmap..."):
                            heatmap = get_gradcam_heatmap(mmodel, img_array, pred_idx)
                        if heatmap is not None:
                            overlaid = overlay_heatmap(image, heatmap)
                            st.image(overlaid, width=400,
                                     caption=f"{mname} attention overlay")
                            hm_resized = cv2.resize(heatmap, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                            hm_colored = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
                            st.image(hm_colored, width=400,
                                     caption=f"{mname} pure heatmap")
                        else:
                            st.warning(f"Could not generate heatmap for {mname}")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation":
    st.subheader("📊 Model Evaluation Metrics")
    
    # Load validation data if available
    val_results_path = os.path.join(config.RESULTS_DIR, "validation_results.json")
    
    if os.path.exists(val_results_path):
        with open(val_results_path, "r") as f:
            val_results = json.load(f)
        
        y_true = np.array(val_results["y_true"])
        y_pred = np.array(val_results["y_pred"])
        y_pred_proba = np.array(val_results["y_pred_proba"])
        class_names = val_results["class_names"]
        
        tab1, tab2, tab3 = st.tabs(["📈 Confusion Matrix", "📉 ROC Curves", "📊 Per-Class Metrics"])
        
        with tab1:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title('Confusion Matrix', fontsize=16, color='#06b6d4')
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Accuracy", f"{(y_true == y_pred).mean():.2%}")
            with col2:
                st.metric("Total Samples", len(y_true))
        
        with tab2:
            st.markdown("### ROC Curves")
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})', linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curves - All Classes', fontsize=16, color='#06b6d4')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Per-Class Performance")
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            
            metrics_data = []
            for cls in class_names:
                metrics_data.append({
                    "Class": cls,
                    "Precision": f"{report[cls]['precision']:.3f}",
                    "Recall": f"{report[cls]['recall']:.3f}",
                    "F1-Score": f"{report[cls]['f1-score']:.3f}",
                    "Support": report[cls]['support']
                })
            
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("### Overall Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Macro Avg Precision", f"{report['macro avg']['precision']:.3f}")
            with col2:
                st.metric("Macro Avg Recall", f"{report['macro avg']['recall']:.3f}")
            with col3:
                st.metric("Macro Avg F1-Score", f"{report['macro avg']['f1-score']:.3f}")
    
    else:
        st.warning("⚠️ No validation results found. Run evaluation first.")
        st.info("""
        To generate evaluation metrics:
        1. Train models using `kaggle_train.py` on Kaggle GPU
        2. The script automatically generates validation results
        3. Download `validation_results.json` from Kaggle output
        4. Place it in the `results/` folder
        """)
        
        if st.button("📝 Generate Sample Evaluation"):
            st.info("This would run evaluation on your validation set. Feature coming soon!")

st.markdown("---")
st.caption("🔬 AI-powered ocular disease detection | Built with TensorFlow & Streamlit")
