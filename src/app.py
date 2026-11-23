
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(
    page_title="Lunar Boulder Hazard Mapper",
    page_icon="🌕",
    layout="wide"
)

st.title("🌕 Lunar Boulder Hazard Mapper")
st.write("AI-powered lunar boulder detection and segmentation")

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

if not MODEL_PATH.exists():
    st.error("Trained model not found. Check models/best.pt.")
    st.stop()

model = load_model()

st.success("AI model loaded successfully!")

uploaded_file = st.file_uploader(
    "Upload a lunar surface image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.295,
    step=0.005
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Lunar Image")
    st.image(image, width="stretch")

    if st.button("Detect Boulders"):
        with st.spinner("Analyzing lunar surface..."):
            results = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                verbose=False
            )

            result = results[0]
            annotated = result.plot(labels=False, boxes=False)
            annotated_image = Image.fromarray(annotated[:, :, ::-1])

        st.subheader("Boulder Detection Results")
        st.image(annotated_image, width="stretch")

        count = len(result.boxes) if result.boxes is not None else 0

        st.metric("Detected Boulders", count)

        st.info(
            "This is a preliminary boulder-hazard assessment, "
            "not a complete lunar landing safety evaluation."
        )
        st.subheader("Adaptive Landing-Zone Intelligence")

        height, width = image.size[1], image.size[0]
        zone_height = height / 3
        zone_width = width / 3

        zone_counts = np.zeros((3, 3), dtype=int)

        if result.masks is not None:
            for polygon in result.masks.xy:
                if len(polygon) == 0:
                    continue

                center_x = np.mean(polygon[:, 0])
                center_y = np.mean(polygon[:, 1])

                col = min(int(center_x / zone_width), 2)
                row = min(int(center_y / zone_height), 2)

                zone_counts[row, col] += 1

        st.write("Candidate boulder detections in each zone:")

        for row in range(3):
            cols = st.columns(3)

            for col in range(3):
                count_in_zone = zone_counts[row, col]

                with cols[col]:
                    st.metric(
                        f"Zone {row + 1}-{col + 1}",
                        int(count_in_zone)
                    )

        minimum_count = np.min(zone_counts)
        lowest_zones = np.argwhere(zone_counts == minimum_count)

        zone_names = [
            f"Zone {row + 1}-{col + 1}"
            for row, col in lowest_zones
        ]

        st.success(
            f"Lowest detected-boulder count: {minimum_count}"
        )

        st.write("Zones with the lowest count:")

        for zone in zone_names:
            st.write(f"- {zone}")

        st.warning(
            "This identifies the zone with the fewest detected candidates, not a confirmed safe landing site."
        )
        st.subheader("Visual Lunar Hazard Map")

        hazard_map = image.copy()
        draw = np.array(hazard_map)

        minimum = np.min(zone_counts)
        maximum = np.max(zone_counts)

        for row in range(3):
            for col in range(3):
                count = zone_counts[row, col]

                if count == minimum:
                    color = (0, 255, 0)
                elif count == maximum:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 0)

                x1 = int(col * zone_width)
                y1 = int(row * zone_height)
                x2 = int((col + 1) * zone_width)
                y2 = int((row + 1) * zone_height)

                cv2.rectangle(
                    draw,
                    (x1, y1),
                    (x2, y2),
                    color,
                    5
                )

                cv2.putText(
                    draw,
                    f"{row + 1}-{col + 1}: {count}",
                    (x1 + 15, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        st.image(draw, caption="Lunar Boulder Density by Zone", width="stretch")