# Precision Object Counting

A sophisticated computer vision application for accurate detection and counting of objects and people in images. This project leverages advanced image processing techniques and contour analysis to provide precise counts with visual feedback and step-by-step processing visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Core Components](#core-components)
- [Processing Pipeline](#processing-pipeline)
- [Configuration Parameters](#configuration-parameters)
- [Output Examples](#output-examples)
- [Dependencies](#dependencies)
- [Docker Support](#docker-support)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Precision Object Counting is an intelligent image analysis application that automatically detects and counts objects and people in images using advanced computer vision algorithms. The application provides:

- **Real-time Detection**: Upload images and get instant object and people counts
- **Visual Feedback**: See exactly where objects and people are detected with visual markers
- **Processing Stages**: View intermediate processing steps to understand the detection pipeline
- **Smart Classification**: Automatically distinguishes between people and objects based on size and aspect ratio characteristics
- **Interactive Web Interface**: User-friendly Streamlit-based UI for easy interaction

The application is particularly useful for:
- Crowd counting in surveillance systems
- Inventory management and object counting
- Automated monitoring systems
- Research in computer vision and image analysis
- Capacity planning and people flow analysis

## ✨ Features

### Core Features

1. **Object Detection and Counting**
   - Automatic detection of objects in images
   - Precise count of detected objects
   - Filtering to reduce false positives

2. **People Detection and Counting**
   - Specialized detection for human figures
   - Distinction between people and objects
   - Head-focused detection for accurate head counts

3. **Image Processing Pipeline**
   - Gaussian blur for noise reduction
   - Otsu's thresholding for binary image creation
   - Morphological operations (dilation and erosion) for contour refinement
   - Adaptive histogram equalization for contrast enhancement
   - Gamma correction for brightness adjustment

4. **Visualization Tools**
   - Real-time processing stage visualization
   - Bounding boxes around detected objects
   - Color-coded markers (green for people, red for objects)
   - Step-by-step pipeline view

5. **Image Manipulation**
   - Image cropping and resizing capabilities
   - Brightness and contrast adjustment
   - Noise reduction filters

6. **User-Friendly Interface**
   - Web-based Streamlit application
   - Easy image upload functionality
   - Real-time processing feedback
   - Interactive parameter selection

### Advanced Features

- **Smart Contour Filtering**: Reduces false positives using size and shape constraints
- **Aspect Ratio Analysis**: Classifies objects based on physical characteristics
- **Adaptive Processing**: Adjusts parameters based on image characteristics
- **Docker Support**: Containerized deployment for consistency

## 📁 Project Structure

```
Precision-Object-Counting/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
├── Images/                         # Sample test images
│   ├── seq_000001.jpg
│   ├── seq_000002.jpg
│   └── ... (additional test images)
├── Major Project Code.ipynb        # Jupyter notebook with detailed code
└── Major Project Documentation.docx # Detailed project documentation
```

## 🏗️ Technical Architecture

### Image Processing Pipeline

The application follows a structured pipeline for accurate detection:

```
Input Image (Grayscale)
    ↓
Gaussian Blur (7x7 kernel)
    ↓
Otsu's Thresholding
    ↓
Morphological Operations (Dilation & Erosion)
    ↓
Contour Detection
    ↓
Size-based Filtering
    ↓
Aspect Ratio Filtering
    ↓
Classification (Object vs People)
    ↓
Visualization & Counting
```

### Key Technologies

- **OpenCV**: Image processing and computer vision operations
- **NumPy**: Numerical computations and array operations
- **Streamlit**: Interactive web interface creation
- **PIL**: Image format handling
- **Matplotlib**: Data visualization and plotting

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chakrapani2122/Precision-Object-Counting.git
   cd Precision-Object-Counting
   ```

2. **Create and activate virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import cv2; import streamlit; print('Installation successful!')"
   ```

### Docker Installation

1. **Build Docker image**
   ```bash
   docker build -t precision-object-counting .
   ```

2. **Run Docker container**
   ```bash
   docker run -p 8501:8501 precision-object-counting
   ```

3. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - The application will automatically open in your default browser at `http://localhost:8501`
   - If not, manually navigate to that address

3. **Upload an image**
   - Click on "Choose an image file" in the sidebar
   - Select a JPG, JPEG, or PNG image from your computer
   - The image must contain objects or people to be detected

4. **View results**
   - The original uploaded image is displayed
   - Select a processing stage from the dropdown menu to see intermediate steps
   - Final counts for objects and people are displayed at the bottom

### Step-by-Step Usage Guide

1. **Upload Phase**
   - Navigate to the "Upload Image" section
   - Click the file uploader and select your image
   - Wait for the image to upload and be displayed

2. **Visualization Phase**
   - Select "Original" to see the input image
   - Select "Blurred" to see the Gaussian blur preprocessing
   - Select "Thresholded" to see the binary image after Otsu's thresholding
   - Select "Dilated and Eroded" to see the morphological operations result
   - Select "People and Objects Marked" to see the final detection with visual markers

3. **Result Interpretation**
   - Green circles indicate detected people
   - Red circles indicate detected objects
   - The success messages show the final counts
   - Adjust image or try different images for better results

## 🔧 Core Components

### Image Processing Functions

#### `otsu_threshold(image)`
Applies Otsu's automatic thresholding with Gaussian blur for noise reduction.
- **Input**: Grayscale image
- **Output**: Binary image with objects/people separated from background
- **Parameters**: 7x7 Gaussian kernel
- **Purpose**: Converts grayscale to binary for easier contour detection

#### `dilation_erosion(image)`
Applies morphological operations to refine object boundaries and remove noise.
- **Input**: Binary image
- **Output**: Refined binary image with better-defined contours
- **Operations**: 
  - Dilation (2 iterations) with 5x5 kernel
  - Erosion (1 iteration) with 5x5 kernel
- **Purpose**: Closes small holes and separates touching objects

#### `filter_contours_by_size(contours, min_area, max_area)`
Filters detected contours based on their area to reduce false positives.
- **Input**: List of contours, min area, max area
- **Output**: Filtered list of relevant contours
- **Default Parameters**: min_area=500, max_area=5000
- **Purpose**: Removes noise artifacts and very large regions

#### `filter_contours_by_size_and_shape(contours, min_area, max_area, min_aspect_ratio, max_aspect_ratio)`
Advanced filtering using both size and shape characteristics.
- **Input**: Contours, area range, aspect ratio range
- **Output**: Filtered contours matching criteria
- **Default Parameters**:
  - min_area=1000, max_area=10000
  - min_aspect_ratio=0.3, max_aspect_ratio=1.0
- **Purpose**: More accurate detection by combining multiple criteria

#### `classify_contours(contours)`
Classifies detected contours as objects or people based on characteristics.
- **People Classification Criteria**:
  - Area: 2000 to 10000 pixels
  - Aspect ratio: 0.3 to 0.8
- **Objects Classification Criteria**:
  - Area: 1000 to 8000 pixels
  - Any aspect ratio
- **Output**: Separate lists of people and object contours

#### `process_image(image)`
Main processing function that orchestrates the entire pipeline.
- **Input**: Grayscale image
- **Output**: Tuple (object_count, people_count)
- **Process**: Applies all preprocessing, detection, and classification steps
- **Error Handling**: Returns (0, 0) on processing errors

### Visualization Functions

#### `draw_bounding_boxes(image, contours, color)`
Draws rectangular bounding boxes around detected contours.
- **Purpose**: Visual representation of detected regions
- **Customizable**: Box color and thickness

#### `draw_head_bounding_boxes(image, contours, color)`
Specialized drawing for people heads.
- **Head Proportion**: Top 30% of bounding box
- **Purpose**: Focus on head region for people detection

#### `draw_circles_on_color_image(image, people_contours, object_contours)`
Marks detected items with colored circles.
- **People**: Green circles on face area
- **Objects**: Red circles at center
- **Radius**: Adaptive based on contour size
- **Output**: Color image with visual markers

#### `get_processing_stages(image)`
Generates images at each step of the processing pipeline.
- **Stages**:
  1. Original
  2. Blurred
  3. Thresholded
  4. Dilated and Eroded
- **Purpose**: Educational visualization and debugging

#### `get_processing_stages_with_circles(image)`
Extended visualization including final detection marks.
- **Includes**: All stages above plus "People and Objects Marked"
- **Purpose**: Complete pipeline visualization

### Enhancement Functions

#### `gamma_correction(image, gamma)`
Adjusts image brightness using gamma correction.
- **Formula**: I_out = (I_in/255)^(1/gamma) * 255
- **Use**: Dark or overexposed images
- **Range**: gamma > 1 (brightens), gamma < 1 (darkens)

#### `adaptive_histogram_equalization(image)`
Enhances local contrast using CLAHE.
- **Parameters**: Clip limit 2.0, 8x8 tile grid
- **Purpose**: Better contrast in varied lighting conditions

#### `resize(image, width, height)`
Resizes image to specified dimensions.
- **Purpose**: Normalize image size for consistent processing

#### `crop(image, start_y, end_y, start_x, end_x)`
Crops image to region of interest.
- **Format**: [rows, columns] indexing
- **Purpose**: Focus on specific areas of images

## 📊 Processing Pipeline

### Detailed Pipeline Explanation

1. **Input Image Loading**
   - Image is loaded as grayscale for faster processing
   - Format support: JPG, JPEG, PNG

2. **Gaussian Blur**
   - Kernel size: 7x7 pixels
   - Purpose: Reduce image noise and smooth transitions
   - Effect: Reduces small artifacts that could be false detections

3. **Otsu's Thresholding**
   - Automatic threshold determination
   - Converts to pure binary (black and white)
   - Purpose: Separate objects from background

4. **Morphological Operations**
   - **Dilation**: Expands white regions (2 iterations)
   - **Erosion**: Shrinks white regions (1 iteration)
   - Kernel: 5x5 square
   - Purpose: Close holes in objects and refine boundaries

5. **Contour Detection**
   - Finds external contours in binary image
   - Contour approximation: CHAIN_APPROX_SIMPLE
   - Output: All detected contours

6. **First-Level Filtering**
   - Area-based filtering: 500-5000 pixels
   - Purpose: Remove very small noise or very large regions

7. **Advanced Filtering**
   - Size range: 1000-10000 pixels
   - Aspect ratio range: 0.3-1.0
   - Purpose: More precise object identification

8. **Classification**
   - Separates people from objects
   - People: 2000-10000 area, 0.3-0.8 aspect ratio
   - Objects: 1000-8000 area
   - Returns separate counts

9. **Visualization**
   - Marks detected items with colored circles
   - Green for people, red for objects
   - Displays final counts

## ⚙️ Configuration Parameters

### Tunable Parameters

The following parameters can be adjusted in `app.py` to optimize for different image types:

**Gaussian Blur Kernel**
```python
cv2.GaussianBlur(image, (7, 7), 0)
```
- Current: 7x7
- Increase for: More blur, less noise
- Decrease for: More detail preservation

**Morphological Kernel**
```python
kernel = np.ones((5, 5), np.uint8)
```
- Current: 5x5
- Increase for: Better contour connection
- Decrease for: Less dilation/erosion effect

**Dilation and Erosion Iterations**
```python
dilated = cv2.dilate(image, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)
```
- Dilation iterations: 2
- Erosion iterations: 1
- Adjust to control boundary refinement

**Contour Filtering Ranges**
```python
filter_contours_by_size_and_shape(contours, 
                                  min_area=1000, 
                                  max_area=10000,
                                  min_aspect_ratio=0.3, 
                                  max_aspect_ratio=1.0)
```
- Area range: 1000-10000 pixels
- Aspect ratio: 0.3-1.0
- Adjust based on target object sizes

**Classification Thresholds**
```python
# People
if 2000 <= area <= 10000 and 0.3 <= aspect_ratio <= 0.8:
    people.append(contour)
    
# Objects
elif 1000 <= area <= 8000:
    objects.append(contour)
```
- People area: 2000-10000
- People aspect ratio: 0.3-0.8
- Object area: 1000-8000

**CLAHE Parameters**
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```
- Clip limit: 2.0 (controls contrast)
- Tile grid: 8x8 (local region size)

## 📸 Output Examples

### Processing Stages

1. **Original Image**: Raw input from user
2. **Blurred Image**: After Gaussian blur to reduce noise
3. **Thresholded Image**: Binary image showing object regions
4. **Dilated and Eroded**: Refined contours after morphological operations
5. **People and Objects Marked**: Final output with green (people) and red (objects) markers

### Count Output

```
Objects Counted: 42
People Counted: 15
```

## 📦 Dependencies

### Python Packages

```
opencv-python-headless==4.8.0.76  # Computer vision operations
numpy==1.24.4                      # Numerical computing
streamlit==1.25.0                  # Web interface framework
matplotlib==3.8.0                  # Plotting and visualization
Pillow                             # Image handling (implicit dependency)
```

### System Dependencies (for Docker)

```
libgl1-mesa-glx  # OpenGL library for headless systems
libglib2.0-0     # GLib library for runtime
```

## 🐳 Docker Support

### Dockerfile Features

- **Base Image**: Python 3.11-slim (lightweight)
- **System Dependencies**: Includes required libraries for headless OpenCV
- **Port**: 8501 (Streamlit default)
- **Address**: 0.0.0.0 (accessible from host)

### Docker Deployment

**Development Mode**
```bash
docker build -t precision-object-counting .
docker run -p 8501:8501 precision-object-counting
```

**Production Mode with Volume**
```bash
docker run -p 8501:8501 -v $(pwd)/Images:/app/Images precision-object-counting
```

**Background Execution**
```bash
docker run -d -p 8501:8501 --name poc precision-object-counting
docker logs -f poc
```

## ⚡ Performance Considerations

### Speed Optimization

1. **Image Size**: Smaller images process faster
   - Recommended: < 1920x1080 pixels
   - Processing time: ~1-3 seconds per image

2. **Parameter Tuning**
   - Smaller area ranges: Faster filtering
   - Fewer dilation iterations: Quicker morphological ops
   - Smaller blur kernel: Faster convolution

3. **Memory Usage**
   - Minimal memory footprint
   - Suitable for edge devices
   - Streamlit handles web session management

### Accuracy Optimization

1. **Image Quality**
   - Good lighting improves detection
   - Clear separation between objects and background
   - Minimal shadows and reflections

2. **Parameter Adjustment**
   - Match area ranges to target objects
   - Adjust aspect ratios for shape variation
   - Increase iterations for complex backgrounds

3. **Preprocessing**
   - Gamma correction for poor lighting
   - Adaptive histogram equalization for contrast
   - Appropriate blur levels for noise reduction

## 🔧 Troubleshooting

### Common Issues

**Issue: No objects detected**
- Solution: Check image quality and lighting
- Try: Increase blur kernel size or adjust thresholds
- Verify: Objects are distinct from background

**Issue: False positives (too many detections)**
- Solution: Increase min_area or narrow aspect ratio range
- Try: Increase erosion iterations
- Verify: Parameter values match target objects

**Issue: False negatives (missed objects)**
- Solution: Decrease min_area or widen aspect ratio range
- Try: Increase dilation iterations
- Verify: Objects meet size criteria

**Issue: People and objects confused**
- Solution: Adjust aspect ratio thresholds
- Try: Fine-tune classification criteria
- Verify: Human proportions in training data

**Issue: Slow performance**
- Solution: Reduce image size
- Try: Decrease blur kernel size
- Verify: Processing stages are necessary

**Issue: Blurry output images**
- Solution: Check input image quality
- Try: Reduce blur kernel size
- Verify: Image is not already blurred

### Installation Troubleshooting

**OpenCV issues (Windows)**
```bash
pip install opencv-python-headless --upgrade
```

**Missing system libraries (Linux)**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

**Streamlit port already in use**
```bash
streamlit run app.py --server.port 8502
```

## 🚀 Future Enhancements

### Planned Features

1. **Deep Learning Integration**
   - YOLO for improved detection
   - Faster R-CNN for complex scenes
   - Convolutional Neural Networks for classification

2. **Advanced Features**
   - Multiple objects tracking across frames
   - Video processing support
   - Real-time camera feed processing
   - Batch processing for multiple images

3. **User Interface Improvements**
   - Parameter tuning sliders
   - Confidence thresholds display
   - Export results as CSV/JSON
   - Custom ROI (Region of Interest) selection

4. **Performance Enhancements**
   - GPU acceleration support
   - Multi-threading for faster processing
   - Caching for repeated images
   - Optimized algorithm selection

5. **Integration Features**
   - API endpoint for programmatic access
   - Database integration for results logging
   - Email notifications for large detections
   - Cloud deployment support

6. **Analytics**
   - Detection confidence scores
   - Processing time metrics
   - Historical trend analysis
   - Export visualizations

## 👥 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Open a Pull Request

### Guidelines

- Follow PEP 8 Python style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation for changes
- Test with various image types

## 📄 License

This project is open source and available under the MIT License. See LICENSE file for details.

---

**Note**: This application is optimized for educational and research purposes. For production use with critical applications (surveillance, safety), consider additional validation and testing.

**Project Resources**:
- Main Code: `Major Project Code.ipynb`
- Detailed Documentation: `Major Project Documentation.docx`
- Sample Images: Located in `Images/` directory

**Last Updated**: 2026-05-29

For questions or issues, please open an issue in the repository.

