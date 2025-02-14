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


"""
input: image, image_file_name, output_path,
usage: output cr channel, cb channel, cr dist, cb dist
        cr diff, cb diff, cr diff mask, cb diff mask.
"""

def crcb_plots(image, mask, save_option=False, output_path=None, image_file_name=None, plt_show=False):
	"""
	image is in cbcr channel
	"""
	# Extract Cr and Cb channels
	cr_channel = image[:, :, 1]  # Red chrominance
	cb_channel = image[:, :, 2]  # Blue chrominance

	# Apply mask to extract only skin pixels
	cr_skin = np.where(mask, cr_channel, np.nan)  # Keep skin, replace non-skin with NaN
	cb_skin = np.where(mask, cb_channel, np.nan)  # Keep skin, replace non-skin with NaN

	cr_skin_vis = np.where(mask, cr_channel, 0)  # Keep skin, replace non-skin with 0
	cb_skin_vis = np.where(mask, cb_channel, 0)  # Keep skin, replace non-skin with 0

	# Flatten and remove NaNs for histogram
	valid_cr = cr_skin[~np.isnan(cr_skin)].astype(np.uint8)
	valid_cb = cb_skin[~np.isnan(cb_skin)].astype(np.uint8)


	# Create new figures for visualization plots
	fig_cbcr, axes_cbcr = plt.subplots(1, 2, figsize=(12, 6))

	# Display original Cr Channel
	cr_plot = axes_cbcr[0].imshow(cr_skin, cmap="Reds")
	cbar_cr = fig_cbcr.colorbar(cr_plot, ax=axes_cbcr[0], fraction=0.046, pad=0.04)
	axes_cbcr[0].set_title("Original Cr Channel (Red Chrominance)")
	cbar_cr.set_label("Cr Intensity")
	axes_cbcr[0].axis("off")

	# Display original Cb Channel
	cb_plot = axes_cbcr[1].imshow(cb_skin, cmap="Blues")
	cbar_cb = fig_cbcr.colorbar(cb_plot, ax=axes_cbcr[1], fraction=0.046, pad=0.04)
	axes_cbcr[1].set_title("Original Cb Channel (Blue Chrominance)")
	cbar_cb.set_label("Cb Intensity")
	axes_cbcr[1].axis("off")

	plt.tight_layout()
	if(plt_show):
		plt.show()
	if(save_option):
			fig_cbcr.savefig(output_path + '/' + image_file_name + '_CrCb.png', dpi=300, bbox_inches='tight')
	plt.close(fig_cbcr)


def crcb_dist_plots(image, mask, save_option=False, output_path=None, image_file_name=None, plt_show=False):

	cr_channel = image[:, :, 1]  # Red chrominance
	cb_channel = image[:, :, 2]  # Blue chrominance

	# Apply mask to extract only skin pixels
	cr_skin = np.where(mask, cr_channel, np.nan)  # Keep skin, replace non-skin with NaN
	cb_skin = np.where(mask, cb_channel, np.nan)  # Keep skin, replace non-skin with NaN

	# Flatten and remove NaNs for histogram
	valid_cr = cr_skin[~np.isnan(cr_skin)].astype(np.uint8)
	valid_cb = cb_skin[~np.isnan(cb_skin)].astype(np.uint8)

	# Find the most frequent value (mode) for both channels
	dom_cr = mode(valid_cr, keepdims=True)[0][0]  # Most frequent Cr value
	dom_cb = mode(valid_cb, keepdims=True)[0][0]  # Most frequent Cb value

	# Plot Cr and Cb distributions
	fig_dist, ax_dist = plt.subplots(1, 2, figsize=(12, 5))

	# Cr Distribution (Red Chrominance)
	hist_cr, bins_cr, _ = ax_dist[0].hist(valid_cr, bins=128, range=[0, 255], color='red', alpha=0.7)
	ax_dist[0].set_title("Cr Distribution (Red Chrominance)")
	ax_dist[0].set_xlabel("Cr Value (0-255)")
	ax_dist[0].set_ylabel("Frequency")

	# Mark mode value on Cr plot
	ax_dist[0].axvline(dom_cr, color='black', linestyle='dashed', linewidth=1.5)
	ax_dist[0].text(dom_cr + 3, max(hist_cr) * 0.9, f'Mode: {dom_cr}', color='black')

	# Cb Distribution (Blue Chrominance)
	hist_cb, bins_cb, _ = ax_dist[1].hist(valid_cb, bins=128, range=[0, 255], color='blue', alpha=0.7)
	ax_dist[1].set_title("Cb Distribution (Blue Chrominance)")
	ax_dist[1].set_xlabel("Cb Value (0-255)")
	ax_dist[1].set_ylabel("Frequency")

	# Mark mode value on Cb plot
	ax_dist[1].axvline(dom_cb, color='black', linestyle='dashed', linewidth=1.5)
	ax_dist[1].text(dom_cb + 3, max(hist_cb) * 0.9, f'Mode: {dom_cb}', color='black')

	plt.tight_layout()
	if(plt_show):
		plt.show()
	if(save_option):
		# Save Cr and Cb Distribution Plot
		fig_dist.savefig(output_path + '/' + image_file_name + '_CrCb_dist.png', dpi=300, bbox_inches='tight')
	plt.close(fig_dist)

	return dom_cr, dom_cb


def crcb_diff_plots(image, mask, dom_cr, dom_cb, save_option=False, output_path=None, image_file_name=None, plt_show=False):
	cr_channel = image[:, :, 1]  # Red chrominance
	cb_channel = image[:, :, 2]  # Blue chrominance

	# Apply mask to extract only skin pixels
	cr_skin = np.where(mask, cr_channel, np.nan)  # Keep skin, replace non-skin with NaN
	cb_skin = np.where(mask, cb_channel, np.nan)  # Keep skin, replace non-skin with NaN

	# Create new figure for difference visualization
	fig_diff, axes_diff = plt.subplots(1, 2, figsize=(12, 6))

	# Display Cr Channel - Dominant Cr
	cr_diff_plot = axes_diff[0].imshow(cr_skin - dom_cr, cmap="Reds")
	cbar_cr_diff = fig_diff.colorbar(cr_diff_plot, ax=axes_diff[0], fraction=0.046, pad=0.04)
	axes_diff[0].set_title("Cr Channel Diff")
	cbar_cr_diff.set_label("Cr Intensity Difference")
	axes_diff[0].axis("off")

	# Display Cb Channel - Dominant Cb
	cb_diff_plot = axes_diff[1].imshow(cb_skin - dom_cb, cmap="Blues")
	cbar_cb_diff = fig_diff.colorbar(cb_diff_plot, ax=axes_diff[1], fraction=0.046, pad=0.04)
	axes_diff[1].set_title("Cb Channel Diff")
	cbar_cb_diff.set_label("Cb Intensity Difference")
	axes_diff[1].axis("off")

	plt.tight_layout()
	if(plt_show):
		plt.show()
	if(save_option):
		fig_diff.savefig(output_path + '/' + image_file_name + '_CrCb_diff.png', dpi=300, bbox_inches='tight')
	plt.close(fig_diff) 


def crcb_diff_mask(image, mask, dom_cr, dom_cb, save_option=False, output_path=None, image_file_name=None, plt_show=False):
	cr_channel = image[:, :, 1]  # Red chrominance
	cb_channel = image[:, :, 2]  # Blue chrominance

	# Apply mask to extract only skin pixels
	cr_skin = np.where(mask, cr_channel, np.nan)  # Keep skin, replace non-skin with NaN
	cb_skin = np.where(mask, cb_channel, np.nan)  # Keep skin, replace non-skin with NaN

	# Create new figure for difference visualization
	img_skin_cr_diff_mask = np.where(cr_skin - dom_cr > 0, cr_skin - dom_cr, 0)
	img_skin_cb_diff_mask = np.where(cb_skin - dom_cb > 0, cb_skin - dom_cb, 0)
	fig_diff_mask, axes_diff_mask = plt.subplots(1, 2, figsize=(12, 6))

	# Find min and max values for proper color scaling
	cr_min, cr_max = np.min(img_skin_cr_diff_mask), np.max(img_skin_cr_diff_mask)
	cb_min, cb_max = np.min(img_skin_cb_diff_mask), np.max(img_skin_cb_diff_mask)

	# Display Cr Channel - Dominant Cr
	cr_diff_plot = axes_diff_mask[0].imshow(img_skin_cr_diff_mask, cmap="Reds", vmin=cr_min, vmax=cr_max)  # Fix colorbar range
	cbar_cr_diff = fig_diff_mask.colorbar(cr_diff_plot, ax=axes_diff_mask[0], fraction=0.046, pad=0.04)
	axes_diff_mask[0].set_title("Cr Channel Diff Mask")
	cbar_cr_diff.set_label("Cr Intensity Difference")
	axes_diff_mask[0].axis("off")

	# Display Cb Channel - Dominant Cb
	cb_diff_plot = axes_diff_mask[1].imshow(img_skin_cb_diff_mask, cmap="Blues", vmin=cb_min, vmax=cb_max)  # Fix colorbar range
	cbar_cb_diff = fig_diff_mask.colorbar(cb_diff_plot, ax=axes_diff_mask[1], fraction=0.046, pad=0.04)
	axes_diff_mask[1].set_title("Cb Channel Diff Mask")
	cbar_cb_diff.set_label("Cb Intensity Difference")
	axes_diff_mask[1].axis("off")

	plt.tight_layout()
	if(plt_show):
		plt.show()
	if(save_option):
		fig_diff_mask.savefig(output_path + '/' + image_file_name + '_CrCb_diff_mask.png', dpi=300, bbox_inches='tight')
	plt.close(fig_diff_mask)
    

	return img_skin_cr_diff_mask, img_skin_cb_diff_mask
