import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def get_centroid(contour):
    """Calculate the centroid of a contour."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def detect_shape(contour):
    """
    Classify the shape of a contour based on vertex count and geometry.
    Returns: shape_name (str)
    """
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    # Classification Logic
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # Check Aspect Ratio for Square vs Rectangle
        shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    else:
        # > 4 vertices: Check Circularity
        area = cv2.contourArea(contour)
        if area > 0:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area
            
            # If high circularity -> Circle, else Polygon
            if circularity > 0.82: # Using 0.82 as threshold for better robustness
                shape = "Circle"
            else:
                shape = "Polygon"
        else:
            shape = "Polygon"
            
    return shape

def process_image(image_np, blur_k, canny_min, canny_max):
    """
    Process the image: Grayscale -> Blur -> Canny -> Dilate/Erode -> Find Contours.
    Returns: processed_image (annotated), data_results (list of dicts)
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Optional: Enhance contrast (helps with dark shapes on dark background)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Gaussian Blur
    # Ensure kernel size is odd
    if blur_k % 2 == 0: blur_k += 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    # 3. Canny Edge Detection
    edged = cv2.Canny(blurred, canny_min, canny_max)
    
    # 4. Dilate/Erode to close gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    # Make a copy for drawing. PIL loads as RGB, so we draw in RGB colors.
    output_image = image_np.copy()
    
    obj_id = 1
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small noise
        if area < 500:
            continue
            
        shape_name = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Get Bounding Box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Draw Contours (Green) - (0, 255, 0) in RGB
        cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
        
        # Draw Bounding Box (Red) - (255, 0, 0) in RGB
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Put Shape Name & Area text
        label = f"{shape_name} {int(area)}"
        # Ensure text does not go off image
        t_y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(output_image, label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        results.append({
            "Object ID": obj_id,
            "Shape Type": shape_name,
            "Area (px)": area,
            "Perimeter (px)": perimeter
        })
        obj_id += 1
        
    return output_image, results

def main():
    st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
    
    # 1. Sidebar Controls
    st.sidebar.title("Settings")
    blur_k = st.sidebar.slider("Gaussian Blur Kernel Size", 3, 15, 5, step=2)
    st.sidebar.subheader("Edge Detection Thresholds")
    canny_min = st.sidebar.slider("Min Threshold", 0, 255, 25)
    canny_max = st.sidebar.slider("Max Threshold", 0, 255, 100)
    
    st.title("Shape & Contour Analyzer")
    
    # 2. Image Input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        # Display Original
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
        # Process Image
        processed_img, data = process_image(image_np, blur_k, canny_min, canny_max)
        
        # Display Processed
        with col2:
            st.image(processed_img, caption="Processed Image", use_container_width=True)
            
        # 6. Output Dashboard Metrics & Table
        st.write("---")
        st.subheader("Analysis Results")
        
        # Metrics
        st.metric("Total Objects Detected", len(data))
        
        # Data Table
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No objects detected. Try adjusting the thresholds.")
            
    else:
        # Warning/Placeholder
        st.warning("Awaiting Image Upload... Please convert your image to JPG/PNG.")
        with col1:
            st.info("Uploaded Image will appear here.")
        with col2:
            st.info("Processed Result will appear here.")

if __name__ == "__main__":
    main()
