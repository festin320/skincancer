import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import yaml

import glob
from natsort import natsorted
from skimage import io, color, measure, filters
from scipy.stats import mode  
import matplotlib.patches as mpatches  # For custom legend
import scipy.ndimage as ndimage
import skimage as ski

from contour_detect import ContourDetector
from blob_detect import BlobDetector
from image_preprocess import Imageprocessor
from process_sat_blue import process_sat_blue
import process_ycrcb as pycrcb
import process_lab as plab


def is_closed(contour, tolerance=2):
    """Check if a contour is closed by comparing first and last point."""
    return np.linalg.norm(contour[0] - contour[-1]) < tolerance

def compute_contour_area(contour):
    return 0.5 * np.abs(np.dot(contour[:, 0], np.roll(contour[:, 1], 1)) - np.dot(contour[:, 1], np.roll(contour[:, 0], 1)))


with open("bd_params_default.yaml", "r") as file:
    params_dict = yaml.safe_load(file)

BLOB = False
PRINT_PARAMS = False
CV2_CONTOUR = False
SCIKIT_CONTOUR = False
FILTER = True

DISPLAY = False
YCRCB = True
LAB = True
SAT_BLUE = False

SAVE_BLOB = False
SAVE_CONTOUR = False
SAVE_DIST = False
SAVE_MASK = True
SAVE_WEIGHTS_AB = False

thres_bars = [5, 10, 15, 20]
color_map = {
                        5: (255, 0, 0),   # Blue
                        10: (0, 255, 0),   # Green
                        15: (0, 0, 255),   # Red
                        20: (255, 255, 0), # Cyan
                    }

weight_a_list = [0.33]
weight_b_list = [0.33]
weight_cr_list= [0.33]
weight_a = 0.33
weight_b = 0.33
weight_cr = 0.33
min_area = 500

# user defined
output_folder = f"contours_wa{weight_a}_wb{weight_b}_wcr{weight_cr}_min{min_area}"

output_root = 'pipeline_output'
output_blob_path = output_root + '/' + 'blob'
output_contour_path = output_root + '/' + 'contour'
output_ycrcb_path = output_root + '/' + 'ycrcb_test'
output_lab_path = output_root + '/' + 'lab_test'
output_lab_contour_path = output_lab_path + 'contour'
output_sat_blue_path = output_root + '/' + 'sat_blue_contours'
output_working_path = output_root + '/' + output_folder


if not os.path.exists(output_blob_path):
    os.makedirs(output_blob_path)
if not os.path.exists(output_contour_path):
    os.makedirs(output_contour_path)
if not os.path.exists(output_ycrcb_path):
    os.makedirs(output_ycrcb_path)
if not os.path.exists(output_lab_path):
    os.makedirs(output_lab_path)
if not os.path.exists(output_sat_blue_path):
    os.makedirs(output_sat_blue_path)
if not os.path.exists(output_working_path):
    os.makedirs(output_working_path)


image_path = 'Doctor Face/Beachkofsky Face Photos/*.JPG'
contour_face_path = 'segmented_images/skin_contours/coords/'
contour_face_suffix = '_skin_contour_coords.npy'

all_image_paths = natsorted(glob.glob(image_path))

for idx in range(len(all_image_paths)):
    image_path = all_image_paths[idx]
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    h, w = image.shape[:2]  # Get image height & width

    coord = np.load(contour_face_path + image_file_name + contour_face_suffix)

    ip = Imageprocessor(image)
    image_bgr = ip.image_bgr
    image_rgb = ip.image_rgb

    # Plot the image
    if(DISPLAY):
        fig = plt.figure()
        plt.imshow(image_rgb)
        plt.scatter(coord[:, 0], coord[:, 1], c='blue', s=1, label="skin contour")  # Scatter plot
        plt.legend()
        plt.title("Face with outline")
        plt.show()
        plt.axis("off")
        plt.close(fig)

    # Create an empty black mask (same size as the image)
    mask_skin = np.zeros((h, w), dtype=np.uint8)
    # Convert coordinates into the required format for OpenCV (list of list of points)
    coord_l = coord.reshape((-1, 1, 2))  # Reshape to OpenCV-compatible format
    cv2.fillPoly(mask_skin, [coord_l], color=255) # inside is 255.
    mask_skin = mask_skin/255 # Binary mask
    mask_skin_3ch = np.repeat(mask_skin[:,:,np.newaxis], 3, axis=2) 
    mask_skin_3ch = mask_skin_3ch.astype(np.uint8) # 3-channel skin layer  

    image_rgb = image_rgb.astype(np.uint8)
    img_skin_rgb = image_rgb * mask_skin_3ch
    img_skin_rgb = img_skin_rgb.astype(np.uint8)

    if(DISPLAY):
        fig = plt.figure()
        plt.imshow(img_skin_rgb)
        plt.scatter(coord[:, 0], coord[:, 1], c='blue', s=1, label="skin contour")  # Scatter plot
        plt.show()
        plt.axis("off")
        plt.close(fig)
    img_skin_bgr = image_bgr * mask_skin_3ch
    img_skin_bgr = img_skin_bgr.astype(np.uint8) # used for blob detection

    if(SAT_BLUE):
        process_sat_blue(image_rgb=img_skin_rgb, mask=mask_skin, image_overlay=image_rgb, output_path=output_sat_blue_path, image_file_name=image_file_name)

    if(YCRCB):
        # prepare input image
        img_skin_ycrcb = cv2.cvtColor(img_skin_rgb, cv2.COLOR_RGB2YCrCb)
        # plot image in each channel
        pycrcb.crcb_plots(img_skin_ycrcb, mask_skin,save_option=False, 
                          output_path=output_ycrcb_path, image_file_name=image_file_name)
        # distribution plot
        dom_cr, dom_cb = pycrcb.crcb_dist_plots(img_skin_ycrcb, mask_skin, save_option=False, 
                                                output_path=None, image_file_name=None)
        pycrcb.crcb_diff_plots(img_skin_ycrcb, mask_skin, dom_cr=dom_cr, dom_cb=dom_cb, 
                               save_option=False, output_path=None, image_file_name=None)
        
        cr_mask, cb_mask = pycrcb.crcb_diff_mask(img_skin_ycrcb, mask_skin, dom_cr=dom_cr, dom_cb=dom_cb, 
                                                 save_option=False, output_path=None, image_file_name=None)


    
        

    if(LAB):
        # prepare input image
        img_skin_lab = cv2.cvtColor(img_skin_rgb, cv2.COLOR_RGB2LAB)
        # plot image in each channel
        plab.lab_plots(img_skin_lab, mask_skin, save_option=False, 
                       output_path=output_lab_path, image_file_name=image_file_name)
        # distribution plot
        dom_a, dom_b = plab.lab_dist_plots(img_skin_lab, mask_skin, save_option=False, 
                                           output_path=None, image_file_name=None)
        
        plab.lab_diff_plots(img_skin_lab, mask_skin, dom_a=dom_a, dom_b=dom_b, 
                            save_option=False, output_path=None, image_file_name=None)
        # disply or save diff mask
        ach_mask, bch_mask = plab.lab_diff_mask(img_skin_lab, mask_skin, dom_a=dom_a, dom_b=dom_b, 
                                                save_option=False, output_path=None, image_file_name=None)



    if(FILTER):
        for weight_a, weight_b, weight_cr in zip(weight_a_list, weight_b_list, weight_cr_list):
            img_skin_ab_diff_sum = ach_mask*weight_a + bch_mask*weight_b + cr_mask*weight_cr
            threshold_ab_avg = (img_skin_ab_diff_sum.min() + img_skin_ab_diff_sum.max()) / 2
            print('threshold', threshold_ab_avg)
            # nonzeros = img_skin_ab_diff_sum[img_skin_ab_diff_sum > 0]
            # threshold_ab_avg = np.median(nonzeros)
            # print('threshold', threshold_ab_avg)
            threshold_ab_l = [threshold_ab_avg]

            plt.figure(figsize=(15, 5))
            contour_overlay = image_rgb.copy()

            for j, threshold_ab in enumerate(threshold_ab_l):
                # Create binary mask
                mask_ab_sum = np.where(img_skin_ab_diff_sum > threshold_ab, 255, 0).astype(np.uint8)
                labeled_array, num_features = ndimage.label(mask_ab_sum)
                # Count pixels in each cluster
                cluster_sizes = ndimage.sum(mask_ab_sum, labeled_array, index=range(1, num_features + 1))
                filtered_mask_ab_sum = np.copy(mask_ab_sum)

                # Iterate through clusters and remove small ones
                for i, size in enumerate(cluster_sizes):
                    if size/255 < min_area:
                        filtered_mask_ab_sum[labeled_array == (i + 1)] = 0  # Set entire cluster to 0
                
                raw_contours = measure.find_contours(filtered_mask_ab_sum, level=0.5)
                filtered_contours = [contour for contour in raw_contours if is_closed(contour)]

                contour_data = {i: contour for i, contour in enumerate(filtered_contours)}


                # Draw contours on the overlay
                for contour in filtered_contours:
                    contour = np.fliplr(contour).astype(np.int32)  # Convert (row, col) to (x, y)
                    cv2.polylines(contour_overlay, [contour], isClosed=True, color=(0, 128, 0), thickness=5)

                # Display each overlay in a subplot
                plt.imshow(contour_overlay)
                plt.axis("off")
                plt.title(f"Threshold: {threshold_ab}")

            # Adjust layout and display/save the figure
            plt.tight_layout()
            if DISPLAY:
                plt.show()
            if SAVE_MASK:
                output_path = f"{output_working_path}/{image_file_name}.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Contour saved at: {output_path}")
                np.save(f"{output_working_path}/{image_file_name}_filtered_contours.npy", contour_data)
            plt.close()

            






                    
 




