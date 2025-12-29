"""
Streamlit Web App for Helmet Detection
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.detector import HelmetDetector
from app.utils.visualizer import Visualizer

# Page config
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
@st.cache_resource
def load_detector():
    """Load detector model (cached)"""
    try:
        detector = HelmetDetector()
        return detector, None
    except Exception as e:
        return None, str(e)

# Initialize detector
detector, error = load_detector()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🪖 Helmet Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if error:
        st.error(f"Error loading model: {error}")
        st.stop()
    
    # Model info
    st.success("✅ Model loaded successfully!")
    st.markdown("### Model Information")
    st.info("**Model:** YOLOv8n\n\n**Classes:**\n- With Helmet\n- Without Helmet\n- Rider\n- Number Plate")
    
    # Detection settings
    st.markdown("### Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detection"
    )
    
    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="IoU threshold for NMS"
    )
    
    if detector:
        detector.conf_threshold = conf_threshold
        detector.iou_threshold = iou_threshold
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("mAP50", "94%")
    st.metric("mAP50-95", "76%")
    st.metric("Precision", "91%")
    st.metric("Recall", "90%")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📸 Image", "🎥 Video", "📹 Webcam", "ℹ️ About"])

# Tab 1: Image Detection
with tab1:
    st.header("Image Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect helmets"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width='stretch')
            
            # Detection button
            if st.button("🔍 Detect", type="primary", width='stretch'):
                with st.spinner("Processing image..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Run detection using YOLO directly
                        yolo_results = detector.model.predict(
                            source=tmp_path,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            save=False,
                            show=False
                        )
                        
                        # Get result
                        r = yolo_results[0]
                        result = detector._parse_results(r)
                        
                        # Get annotated image from YOLO (already has boxes drawn)
                        result_image = r.plot()  # YOLO's built-in plot method
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                        with col2:
                            st.subheader("Detection Results")
                            st.image(result_image, caption="Detected Objects", width='stretch')
                            
                            # Statistics
                            st.markdown("### 📊 Statistics")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Total Detections", result['count'])
                            with col_b:
                                st.metric("Image Size", f"{image.size[0]}x{image.size[1]}")
                            
                            # Detection details
                            if result['count'] > 0:
                                st.markdown("### 🔍 Detection Details")
                                details_data = []
                                for i, (box, label, conf) in enumerate(zip(
                                    result['boxes'],
                                    result['labels'],
                                    result['confidences']
                                ), 1):
                                    details_data.append({
                                        "ID": i,
                                        "Class": label,
                                        "Confidence": f"{conf:.2%}",
                                        "Box": f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
                                    })
                                st.dataframe(details_data, width='stretch', hide_index=True)
                                
                                # Class counts
                                from collections import Counter
                                class_counts = Counter(result['labels'])
                                st.markdown("### 📈 Class Distribution")
                                st.bar_chart(class_counts)
                            else:
                                st.info("No objects detected. Try adjusting the confidence threshold.")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        else:
            st.info("👆 Please upload an image to start detection")

# Tab 2: Video Detection
with tab2:
    st.header("Video Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to detect helmets"
        )
        
        if uploaded_video is not None:
            # Display video info
            st.video(uploaded_video)
            
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                tmp_video_path = tmp_file.name
            
            st.info(f"Video uploaded: {uploaded_video.name}")
            
            # Detection button
            if st.button("🔍 Process Video", type="primary", width='stretch'):
                with st.spinner("Processing video... This may take a while..."):
                    try:
                        # Create output path
                        output_dir = Path("output/videos")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f"result_{uploaded_video.name}"
                        
                        # Process video
                        stats = detector.predict_video(
                            video_path=tmp_video_path,
                            output_path=str(output_path),
                            show=False
                        )
                        
                        with col2:
                            st.subheader("Processing Results")
                            st.success("✅ Video processed successfully!")
                            
                            # Statistics
                            st.markdown("### 📊 Video Statistics")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Frames Processed", stats['frames'])
                            with col_b:
                                st.metric("Total Detections", stats['total_detections'])
                            with col_c:
                                st.metric("Avg/Frame", f"{stats['avg_detections_per_frame']:.2f}")
                            
                            # Download processed video
                            if output_path.exists():
                                st.markdown("### 📥 Download Processed Video")
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download Video",
                                        data=f.read(),
                                        file_name=f"detected_{uploaded_video.name}",
                                        mime="video/mp4",
                                        width='stretch'
                                    )
                        
                        # Clean up
                        os.unlink(tmp_video_path)
                    
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        if os.path.exists(tmp_video_path):
                            os.unlink(tmp_video_path)
        else:
            st.info("👆 Please upload a video to start detection")

# Tab 3: Webcam
with tab3:
    st.header("Real-time Webcam Detection")
    st.info("⚠️ Webcam detection will open a separate window. Press 'q' to quit.")
    
    if st.button("🎥 Start Webcam", type="primary", width='stretch'):
        with st.spinner("Starting webcam..."):
            try:
                detector.predict_webcam(camera_id=0, show=True)
                st.success("Webcam stopped successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure your webcam is connected and not being used by another application.")

# Tab 4: About
with tab4:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application detects motorcycle riders with and without helmets using YOLOv8 deep learning model.
    
    ### 🚀 Features
    - **Image Detection**: Upload and detect helmets in images
    - **Video Detection**: Process videos frame by frame
    - **Real-time Webcam**: Live detection using your webcam
    - **Configurable Thresholds**: Adjust confidence and IoU thresholds
    
    ### 📊 Model Information
    - **Architecture**: YOLOv8n (nano)
    - **Training**: 50 epochs
    - **Performance**:
        - mAP50: 94%
        - mAP50-95: 76%
        - Precision: 91%
        - Recall: 90%
    
    ### 🏷️ Detection Classes
    1. **With Helmet** - Riders wearing helmets
    2. **Without Helmet** - Riders not wearing helmets
    3. **Rider** - Motorcycle riders
    4. **Number Plate** - Vehicle license plates
    
    ### 🛠️ Technology Stack
    - **Framework**: Ultralytics YOLOv8
    - **UI**: Streamlit
    - **Computer Vision**: OpenCV
    - **Python**: 3.8+
    
    ### 📝 Usage Tips
    1. For best results, use images with clear visibility
    2. Adjust confidence threshold if you get too many/few detections
    3. Video processing may take time depending on video length
    4. Ensure good lighting when using webcam
    
    ### 🤝 Support
    For issues or questions, please check the documentation or repository.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Made with <b>AnhCBT</b> using Streamlit and YOLOv8"
    "</div>",
    unsafe_allow_html=True
)

