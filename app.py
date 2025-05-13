import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Function to load the model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Function to perform object detection
def detect_objects(image, model):
    results = model(image)
    return results

# Function to calculate freshness score
def calculate_freshness_score(confidence, class_name):
    base_score = confidence * 100
    
    # Adjust score based on class (assuming 'rotten' classes have lower scores)
    if 'rotten' in class_name.lower():
        base_score *= 0.5
    
    return min(max(int(base_score), 1), 100)  # Ensure score is between 1 and 100

# Function to draw bounding boxes, labels, and freshness scores on the image
def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            class_name = result.names[int(box.cls[0])]
            confidence = box.conf[0]
            freshness_score = calculate_freshness_score(confidence, class_name)
            
            # Determine color based on freshness score
            color = (0, 255, 0) if freshness_score >= 50 else (0, 0, 255)  # Green for fresh, Red for rotten
            
            # Draw bounding box
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), color, 2)
            
            # Draw labels
            label = f"{class_name} ({confidence:.2f})"
            freshness_label = f"Freshness: {freshness_score}/100"
            cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(image, freshness_label, (r[0], r[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image

# Function to calculate and display summary statistics
def display_summary_statistics(results):
    data = {
        'Class': [],
        'Count': [],
        'Average Freshness': []
    }
    
    for result in results:
        class_counts = {}
        freshness_scores = {}
        
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            confidence = box.conf[0]
            freshness_score = calculate_freshness_score(confidence, class_name)
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                freshness_scores[class_name] = []
                
            class_counts[class_name] += 1
            freshness_scores[class_name].append(freshness_score)
        
        for class_name, count in class_counts.items():
            avg_freshness = np.mean(freshness_scores[class_name])
            data['Class'].append(class_name)
            data['Count'].append(count)
            data['Average Freshness'].append(avg_freshness)
    
    df = pd.DataFrame(data)
    
    # Display DataFrame
    st.write("### Summary Statistics")
    st.write(df)
    
    # Plot Count Distribution
    fig_count = go.Figure()
    fig_count.add_trace(go.Bar(x=df['Class'], y=df['Count'], name='Count'))
    fig_count.update_layout(title='Count of Each Fruit Type',
                            xaxis_title='Fruit Type',
                            yaxis_title='Count')
    st.plotly_chart(fig_count)
    
    # Plot Average Freshness Distribution
    fig_freshness = go.Figure()
    fig_freshness.add_trace(go.Bar(x=df['Class'], y=df['Average Freshness'], name='Average Freshness'))
    fig_freshness.update_layout(title='Average Freshness Score by Fruit Type',
                                xaxis_title='Fruit Type',
                                yaxis_title='Average Freshness Score')
    st.plotly_chart(fig_freshness)

# Streamlit app
def main():
    st.title("Fruit Freshness Classifier")
    
    # Load the pre-trained model
    model_path = "best.pt"  # Make sure this file is in the same directory as your script
    model = load_model(model_path)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        
        # Perform object detection
        if st.button("Classify Fruit"):
            results = detect_objects(image, model)
            
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            
            # Draw bounding boxes and freshness scores
            image_with_boxes = draw_boxes(image_np, results)
            
            # Convert image with boxes back to PIL Image for display
            image_with_boxes_pil = Image.fromarray(image_with_boxes)
            
            # Display the result
            st.image(image_with_boxes_pil, caption="Classification Result", use_container_width=True)

            
            # Display detection information
            st.subheader("Classification Information")
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    confidence = box.conf[0]
                    freshness_score = calculate_freshness_score(confidence, class_name)
                    
                    # Color coding for freshness score
                    if freshness_score < 50:
                        color = 'red'
                        classification_note = 'Rotten'
                    else:
                        color = 'green'
                        classification_note = 'Fresh'
                    
                    # Create a color box with bold heading
                    color_box_html = f'<div style="background-color: {color}; width: 50px; height: 20px; display: inline-block; border: 1px solid black;"></div>'
                    st.markdown(f"**Color Box**: {color_box_html} **Color**: {color}", unsafe_allow_html=True)
                    
                    # Classification information
                    st.write(f"**Class**: {class_name}")
                    st.write(f"**Confidence**: {confidence:.2f}")
                    st.write(f"**Freshness Score**: {freshness_score}/100")
                    st.write(f"**Classification**: {classification_note}")
                    st.write("")  # Add a blank line for spacing
            
            # Display summary statistics
            display_summary_statistics(results)

if __name__ == "__main__":
    main()
