import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage

def is_closed(contour, tolerance=2):
    """Check if a contour is closed by comparing first and last point."""
    return np.linalg.norm(contour[0] - contour[-1]) < tolerance

def process_sat_blue(image_rgb, mask, image_overlay, output_path, image_file_name, plt_show=False):
    # Convert to HLS and extract Saturation channel
    image_hls = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
    img_sat = image_hls[:, :, 2]
    # Extract Blue Channel from RGB
    img_blue = image_rgb[:, :, 2]

    img_skin_sat = np.where(mask, img_sat, np.nan)
    img_skin_blue = np.where(mask, img_blue, np.nan)
    valid_skin_sat = img_skin_sat[~np.isnan(img_skin_sat)].astype(np.uint8)
    valid_skin_blue = img_skin_blue[~np.isnan(img_skin_blue)].astype(np.uint8)

    # Compute histogram
    hist_sat, bins_sat = np.histogram(valid_skin_sat.flatten(), bins=256, range=[0,255])
    hist_blue, bins_blue = np.histogram(valid_skin_blue.flatten(), bins=256, range=[0,255])
    mode_sat = bins_sat[np.argmax(hist_sat)]
    mode_blue = bins_blue[np.argmax(hist_blue)]

    img_skin_sat_diff_mask = np.where(img_skin_sat - mode_sat > 0, img_skin_sat - mode_sat, 0)
    # img_skin_blue_diff_mask = np.where(img_skin_blue - mode_blue > 0, img_skin_blue - mode_blue, 0)
    img_skin_blue_diff_mask = np.where(img_skin_blue - mode_blue < 0, -(img_skin_blue - mode_blue), 0)


    # Define thresholds for both channels
    saturation_threshold = np.mean(img_skin_sat_diff_mask) + 120  # Adjust as needed
    blue_threshold = np.mean(img_skin_blue_diff_mask) + 20            # Adjust as needed

    if(img_skin_sat_diff_mask.max() < 1):
        return

    # Create binary masks
    mask_saturation = np.where(img_skin_sat_diff_mask > saturation_threshold, 255, 0).astype(np.uint8)
    mask_blue = np.where(img_skin_blue_diff_mask > blue_threshold, 255, 0).astype(np.uint8)

    # Label clusters in binary masks
    labeled_saturation, num_saturation = ndimage.label(mask_saturation)
    labeled_blue, num_blue = ndimage.label(mask_blue)

    # Count pixels in each cluster
    cluster_sizes_saturation = ndimage.sum(mask_saturation, labeled_saturation, index=range(1, num_saturation + 1))
    cluster_sizes_blue = ndimage.sum(mask_blue, labeled_blue, index=range(1, num_blue + 1))

    print('len(sat): ', (num_saturation))
    print('len(blue): ', (num_blue))

    # Define minimum area threshold
    min_area = 0

    # Remove small clusters
    filtered_mask_saturation = np.copy(mask_saturation)
    filtered_mask_blue = np.copy(mask_blue)

    for i, size in enumerate(cluster_sizes_saturation):
        if size / 255 < min_area:
            filtered_mask_saturation[labeled_saturation == (i + 1)] = 0

    for i, size in enumerate(cluster_sizes_blue):
        if size / 255 < min_area:
            filtered_mask_blue[labeled_blue == (i + 1)] = 0

    # Find and filter contours
    raw_contours_saturation = measure.find_contours(filtered_mask_saturation, level=0.5)
    raw_contours_blue = measure.find_contours(filtered_mask_blue, level=0.5)

    filtered_contours_saturation = [contour for contour in raw_contours_saturation if is_closed(contour)]
    filtered_contours_blue = [contour for contour in raw_contours_blue if is_closed(contour)]

    # Create overlay images
    contour_overlay_saturation = image_overlay.copy()
    contour_overlay_blue = image_overlay.copy()


    # Draw contours
    for contour in filtered_contours_saturation:
        contour = np.fliplr(contour).astype(np.int32)
        cv2.polylines(contour_overlay_saturation, [contour], isClosed=True, color=(255, 0, 0), thickness=3)

    for contour in filtered_contours_blue:
        contour = np.fliplr(contour).astype(np.int32)
        cv2.polylines(contour_overlay_blue, [contour], isClosed=True, color=(0, 255, 255), thickness=3)


    # Plot both saturation and blue channel contours
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(contour_overlay_saturation)
    # sat_plot = axes[0].imshow(img_skin_sat_diff_mask, cmap="viridis")
    # fig.colorbar(sat_plot, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar
    axes[0].axis("off")
    axes[0].set_title("Contours on Saturation Channel")
    
    # axes[1].imshow(contour_overlay_blue)
    blue_plot = axes[1].imshow(img_skin_blue_diff_mask, cmap="Blues", vmax=15)
    fig.colorbar(blue_plot, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar
    axes[1].axis("off")
    axes[1].set_title("Contours on Blue Channel")
    

    plt.tight_layout()
    if(plt_show):
        plt.show()

    # Save the figure if needed
    output_path = f"{output_path}/{image_file_name}_sat_blue_contours.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Contour saved at: {output_path}")
    plt.close(fig)






