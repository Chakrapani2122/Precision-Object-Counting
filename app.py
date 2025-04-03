import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ...existing functions (plotting_grid)...

def dilation_erosion(image):
    """Apply optimized dilation and erosion to refine object boundaries."""
    kernel = np.ones((5, 5), np.uint8)  # Increased kernel size for better contour refinement
    dilated = cv2.dilate(image, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def otsu_threshold(image):
    """Apply Otsu's thresholding with Gaussian blur for noise reduction."""
    blurred = cv2.GaussianBlur(image, (7, 7), 0)  # Increased kernel size for better noise reduction
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

def filter_contours_by_size_and_shape(contours, min_area=1000, max_area=10000, min_aspect_ratio=0.3, max_aspect_ratio=1.0):
    """Filter contours based on size and aspect ratio to reduce false positives."""
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            filtered_contours.append(contour)
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
        elif 1000 <= area <= 8000:
            objects.append(contour)
    return objects, people

def process_image(image):
    """Process the image and count objects and people."""
    try:
        # Preprocessing
        thresholded_img = otsu_threshold(image)
        dilated_eroded_img = dilation_erosion(thresholded_img)
        
        # Contour detection and filtering
        contours, _ = cv2.findContours(dilated_eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_contours_by_size_and_shape(contours, min_area=1000, max_area=10000)
        
        # Classify contours
        objects, people = classify_contours(filtered_contours)
        
        # Return counts
        return len(objects), len(people)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return 0, 0

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

def draw_circles(image, people_contours, object_contours):
    """Draw small green circles for people and red circles for objects."""
    output_image = image.copy()
    for contour in people_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        radius = 10  # Small circle radius
        cv2.circle(output_image, center, radius, (0, 255, 0), -1)  # Green circle for people
    for contour in object_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        radius = 10  # Small circle radius
        cv2.circle(output_image, center, radius, (0, 0, 255), -1)  # Red circle for objects
    return output_image

def draw_circles_on_color_image(image, people_contours, object_contours):
    """Draw green circles for people's faces and red circles for objects on a color image."""
    # Convert grayscale image to BGR to allow colored circles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in people_contours:
        x, y, w, h = cv2.boundingRect(contour)
        face_center = (x + w // 2, y + int(h * 0.15))  # Adjusted to place the circle on the face (top 15% of the bounding box)
        face_radius = max(w, h) // 8  # Adjusted radius for better accuracy
        cv2.circle(output_image, face_center, face_radius, (0, 255, 0), -1)  # Green circle for people's faces
    for contour in object_contours:
        x, y, w, h = cv2.boundingRect(contour)
        object_center = (x + w // 2, y + h // 2)  # Center of the bounding box for objects
        object_radius = min(w, h) // 4  # Adjusted radius for objects
        cv2.circle(output_image, object_center, object_radius, (0, 0, 255), -1)  # Red circle for objects
    return output_image

def get_processing_stages_with_circles(image):
    """Return images at different processing stages, including one with circles."""
    stages = get_processing_stages(image)
    thresholded_img = otsu_threshold(image)
    dilated_eroded_img = dilation_erosion(thresholded_img)
    contours, _ = cv2.findContours(dilated_eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours_by_size_and_shape(contours, min_area=1000, max_area=10000)
    objects, people = classify_contours(filtered_contours)
    stages["People and Objects Marked"] = draw_circles_on_color_image(image, people, objects)
    return stages

# Streamlit app
st.title("Object and People Counting App")

# Remove camera option, keep only the upload image option
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the uploaded image directly using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_np = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    st.image(image_np, caption="Uploaded Image", use_column_width=True)
    
    # Get processing stages with circles
    stages = get_processing_stages_with_circles(image_np)
    stage_names = ["No Image Selected"] + list(stages.keys())
    
    # Dropdown to select processing stage
    selected_stage = st.selectbox("Select Processing Stage", stage_names)
    if selected_stage != "No Image Selected":
        st.image(stages[selected_stage], caption=f"{selected_stage} Image", use_column_width=True)
    else:
        st.write("No image selected for display.")
    
    # Final processing and count
    st.write("Processing...")
    object_count, people_count = process_image(image_np)
    st.success(f"Objects Counted: {object_count}")
    st.success(f"People Counted: {people_count}")
