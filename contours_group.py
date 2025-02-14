#%%

import os
import copy
import numpy as np
import cv2
import yaml
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN

def group_contours_by_distance(contours, max_distance=1024):
    """
    Groups contours based on their proximity using DBSCAN.
    
    Parameters:
    - contours: List of numpy arrays, each representing contour points.
    - max_distance: Maximum allowed distance between contour groups.
    
    Returns:
    - grouped_contours: List of lists, where each sublist contains contours belonging to the same group.
    - bounding_boxes: List of bounding boxes for each grouped contour set.
    """
    
    # Get centroid of each contour
    centroids = np.array([np.mean(contour, axis=0) for contour in contours])

    # Apply DBSCAN clustering based on centroid distance
    clustering = DBSCAN(eps=max_distance, min_samples=1).fit(centroids)
    labels = clustering.labels_

    # Group contours based on DBSCAN labels
    grouped_contours = {}
    for label, contour in zip(labels, contours):
        if label not in grouped_contours:
            grouped_contours[label] = []
        grouped_contours[label].append(contour)

    # Compute bounding boxes for each group
    bounding_boxes = []
    for group in grouped_contours.values():
        all_points = np.vstack(group)  # Merge all contour points
        all_points = np.array(all_points, dtype=np.int32)  # Convert to int32
        if all_points is None or len(all_points) == 0:
            print("Error: No points found for bounding rect.")
        else:
            x, y, w, h = cv2.boundingRect(all_points)
            w = max(w, 1024)
            h = max(h, 1024)

        
        # Ensure bounding box fits within 1024x1024
        if w <= 1024 and h <= 1024:
            bounding_boxes.append((x, y, w, h))


    return list(grouped_contours.values()), bounding_boxes







#%%
folder = "pipeline_output/contours_wa0.33_wb0.33_wcr0.33_min500"
idx = 5

image_path = 'pipeline_output/contours_wa0.33_wb0.33_wcr0.33_min500/*.npy'
all_image_paths = natsorted(glob.glob(image_path))


image_path = all_image_paths[idx]
image_file_name = os.path.splitext(os.path.basename(image_path))[0]
print(image_file_name)

loaded_contours = np.load(f"{folder}/{image_file_name}.npy", allow_pickle=True).item()

# Access individual contours
# for index, contour in loaded_contours.items():
#     print(f"Contour {index}: {contour.shape} points")

image_height = 6000  # Change this to match your original image height
image_width = 4000   # Change this to match your original image width


# Create a blank black image
mask = np.zeros((image_height, image_width), dtype=np.uint8)

full_contours = []
for key, contour in loaded_contours.items():
	contour = np.fliplr(contour)
	full_contours.append(contour)
	contour = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))  # Reshape to (N, 1, 2)
	cv2.fillPoly(mask, [contour], 128)  # Fill contour with white
	

# Example usage
grouped_contours, bounding_boxes = group_contours_by_distance(full_contours, max_distance=256)


for box in bounding_boxes:
    x, y, w, h = box
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, 5)  # Draw white bounding boxes

# Show result
plt.imshow(mask, cmap='gray')
plt.title("Grouped Contours with 1024x1024 Bounding Boxes")
plt.axis()
plt.show()
