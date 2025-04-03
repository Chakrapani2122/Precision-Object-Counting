import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ...existing functions (plotting_grid)...

def dilation_erosion(image):
    """Apply optimized dilation and erosion to refine object boundaries."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def otsu_threshold(image):
    """Apply Otsu's thresholding with Gaussian blur for noise reduction."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def labeling(image):
    """Label objects in the binary image using contours."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def crop(image, start_y, end_y, start_x, end_x):
    """Crop the image to the specified coordinates."""
    return image[start_y:end_y, start_x:end_x]

def resize(image, width, height):
    """Resize the image to the specified width and height."""
    return cv2.resize(image, (width, height))

def gamma_correction(image, gamma):
    """Apply gamma correction to the image."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_histogram_equalization(image):
    """Apply adaptive histogram equalization to enhance contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def draw_bounding_boxes(image, contours, color):
    """Draw bounding boxes around detected contours."""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

def filter_contours_by_size(contours, min_area=500, max_area=5000):
    """Filter contours based on size to reduce false positives."""
    filtered_contours = [contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area]
    return filtered_contours

def classify_contours(contours):
    """Classify contours as objects or people based on size and aspect ratio."""
    objects = []
    people = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Classify as people if area and aspect ratio match typical human shapes
        if 2000 <= area <= 10000 and 0.3 <= aspect_ratio <= 0.8:
            people.append(contour)
        # Classify as objects otherwise
        elif 800 <= area <= 10000:
            objects.append(contour)
    return objects, people

def process_image(image):
    try:
        # Crop, Resize, Gamma Correction
        cropped_images = []
        for (start_y, end_y, start_x, end_x) in [(100, 380, 380, 600), (20, 220, 100, 300), (200, 500, 0, 300)]:
            cropped_img = crop(image, start_y, end_y, start_x, end_x)
            if cropped_img is not None:
                cropped_images.append(cropped_img)
        # Resize, Gamma Correction, Adaptive Histogram Equalization, Otsu's Thresholding, Dilation and Erosion
        dlt_er_img = None
        for j, cropped_img in enumerate(cropped_images):
            resized_img = resize(cropped_img, cropped_img.shape[1] * 2, cropped_img.shape[0] * 2)
            if resized_img is None:
                continue
            gamma_corrected_img = gamma_correction(resized_img, [1.2, 0.5, 2.5][j])
            if gamma_corrected_img is None:
                continue
            blurred_img = cv2.GaussianBlur(gamma_corrected_img, (5, 5), 0)
            equalized_img = adaptive_histogram_equalization(blurred_img)
            if equalized_img is None:
                continue
            thresholded_img = otsu_threshold(equalized_img)
            if thresholded_img is None:
                continue
            dlt_er_img = dilation_erosion(thresholded_img)
            if dlt_er_img is None:
                continue
        # Labeling using contours
        count = labeling(dlt_er_img) if dlt_er_img is not None else 0
        return count
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return 0

def draw_head_bounding_boxes(image, contours, color):
    """Draw bounding boxes around the heads of detected people."""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        head_height = int(h * 0.3)  # Assume the head is the top 30% of the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + head_height), color, 2)

def draw_fixed_size_boxes(image, contours, color, box_size=50):
    """Draw fixed-size squares around detected objects."""
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + box_size, y + box_size), color, 2)

def get_processing_stages(image):
    """Return images at different processing stages."""
    stages = {}
    stages["Original"] = image
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    stages["Blurred"] = blurred_img
    thresholded_img = otsu_threshold(blurred_img)
    stages["Thresholded"] = thresholded_img
    dilated_eroded_img = dilation_erosion(thresholded_img)
    stages["Dilated and Eroded"] = dilated_eroded_img
    return stages

# Streamlit app
st.title("Object and People Counting App")

# Move options to the main page
option = st.radio("Choose an option:", ("Access Camera", "Upload Image"))

# Initialize session state for camera control
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if option == "Access Camera":
    st.subheader("Access Camera")
    
    # Start and Stop Camera buttons
    start_camera = st.button("Start Camera")
    stop_camera = st.button("Stop Camera")

    if start_camera:
        st.session_state.camera_running = True
    if stop_camera:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        if cap.isOpened():
            while True:
                if not st.session_state.camera_running:
                    break
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access the camera.")
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Preprocessing and contour detection
                thresholded_img = otsu_threshold(gray_frame)
                dilated_eroded_img = dilation_erosion(thresholded_img)
                contours, _ = cv2.findContours(dilated_eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter and classify contours
                filtered_contours = filter_contours_by_size(contours, min_area=800, max_area=10000)
                objects, people = classify_contours(filtered_contours)
                
                # Draw bounding boxes for objects and people
                draw_fixed_size_boxes(frame, objects, (0, 0, 255), box_size=50)  # Red fixed-size squares for objects
                draw_head_bounding_boxes(frame, people, (0, 255, 0))  # Green head-only squares for people
                
                # Display counts
                object_count = len(objects)
                people_count = len(people)
                cv2.putText(frame, f"Objects: {object_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"People: {people_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                stframe.image(frame, channels="BGR")
            cap.release()
        else:
            st.error("Unable to access the camera.")

elif option == "Upload Image":
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image.convert("L"))  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Get processing stages
        stages = get_processing_stages(image_np)
        stage_names = ["No Image Selected"] + list(stages.keys())
        
        # Dropdown to select processing stage
        selected_stage = st.selectbox("Select Processing Stage", stage_names)
        if selected_stage != "No Image Selected":
            st.image(stages[selected_stage], caption=f"{selected_stage} Image", use_column_width=True)
        else:
            st.write("No image selected for display.")
        
        # Final processing and count
        st.write("Processing...")
        count = process_image(image_np)
        st.success(f"Objects/People Counted: {count}")
