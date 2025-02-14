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


def lab_plots(image, mask, save_option=False, output_path=None, image_file_name=None, plt_show=False):
    img_skin_ach = image[:,:,1]
    img_skin_bch = image[:,:,2]

    # Create a figure with two subplots
    fig_lab, axes_lab = plt.subplots(1, 2, figsize=(12, 6))

    # Display A Channel (Green-Red)
    a_plot = axes_lab[0].imshow(img_skin_ach, cmap="RdBu_r")  # Using red-blue colormap
    cbar_a = fig_lab.colorbar(a_plot, ax=axes_lab[0], fraction=0.046, pad=0.04)
    axes_lab[0].set_title("A Channel (Green-Red)")
    cbar_a.set_label("Intensity")
    axes_lab[0].axis("off")

    # Display B Channel (Blue-Yellow)
    b_plot = axes_lab[1].imshow(img_skin_bch, cmap="PuOr")  # Using purple-orange colormap
    cbar_b = fig_lab.colorbar(b_plot, ax=axes_lab[1], fraction=0.046, pad=0.04)
    axes_lab[1].set_title("B Channel (Blue-Yellow)")
    cbar_b.set_label("Intensity")
    axes_lab[1].axis("off")

    plt.tight_layout()
    if(plt_show):
        plt.show()
    if(save_option):
        fig_lab.savefig(output_path + '/' + image_file_name + '_LAB.png', dpi=300, bbox_inches='tight')
    plt.close(fig_lab)


def lab_dist_plots(image, mask, save_option=False, output_path=None, image_file_name=None, plt_show=False):
    img_skin_ach = image[:,:,1]
    img_skin_bch = image[:,:,2]

    img_skin_a = np.where(mask, img_skin_ach, np.nan)
    img_skin_b = np.where(mask, img_skin_bch, np.nan)
    # Flatten and remove NaNs for histogram
    valid_skin_a = img_skin_a[~np.isnan(img_skin_a)].astype(np.uint8)
    valid_skin_b = img_skin_b[~np.isnan(img_skin_b)].astype(np.uint8)

    # Get min and max values for dynamic x-limits
    a_min, a_max = valid_skin_a.min(), valid_skin_a.max()
    b_min, b_max = valid_skin_b.min(), valid_skin_b.max()

    # Compute histogram data
    hist_a, bins_a = np.histogram(valid_skin_a.flatten(), bins=256, range=[0, 255])
    hist_b, bins_b = np.histogram(valid_skin_b.flatten(), bins=256, range=[0, 255])

    # Find most frequent values
    dom_a = bins_a[np.argmax(hist_a)]
    dom_b = bins_b[np.argmax(hist_b)]

    # Plot the histograms
    fig_ab_dist, ax_ab_dist = plt.subplots(1, 2, figsize=(12, 5))

    # A Channel Histogram
    ax_ab_dist[0].hist(valid_skin_a.ravel(), bins=256, range=[0, 255], color='green', alpha=0.7)
    ax_ab_dist[0].axvline(dom_a, color='black', linestyle='dashed', linewidth=1.5)  # Mode line
    ax_ab_dist[0].text(dom_a + 5, max(hist_a) * 0.9, f'Mode: {int(dom_a)}', color='black')
    ax_ab_dist[0].set_xlim(a_min, a_max)  # Set dynamic x-axis limits
    ax_ab_dist[0].set_title("A Channel Distribution (Green-Red)")
    ax_ab_dist[0].set_xlabel("A Value")
    ax_ab_dist[0].set_ylabel("Frequency")
    # ax_ab_dist[0].set_yscale("log")

    # B Channel Histogram
    ax_ab_dist[1].hist(valid_skin_b.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
    ax_ab_dist[1].axvline(dom_b, color='black', linestyle='dashed', linewidth=1.5)  # Mode line
    ax_ab_dist[1].text(dom_b + 5, max(hist_b) * 0.9, f'Mode: {int(dom_b)}', color='black')
    ax_ab_dist[1].set_xlim(b_min, b_max)  # Set dynamic x-axis limits
    ax_ab_dist[1].set_title("B Channel Distribution (Blue-Yellow)")
    ax_ab_dist[1].set_xlabel("B Value")
    ax_ab_dist[1].set_ylabel("Frequency")
    # ax_ab_dist[1].set_yscale("log")

    plt.tight_layout()
    if(plt_show):
        plt.show()
    if(save_option):
        fig_ab_dist.savefig(output_path + '/' + image_file_name + '_AB_dist.png', dpi=300, bbox_inches='tight')
    plt.close(fig_ab_dist)

    return dom_a, dom_b


def lab_diff_plots(image, mask, dom_a, dom_b, save_option=False, output_path=None, image_file_name=None, plt_show=False):

    img_skin_ach = image[:,:,1]
    img_skin_bch = image[:,:,2]

    # Create a figure with two subplots for diff plot
    fig_ab_diff, axes_ab_diff = plt.subplots(1, 2, figsize=(12, 6))

    # Display A Channel (Green-Red)
    a_plot = axes_ab_diff[0].imshow(img_skin_ach - dom_a, cmap="RdBu_r")  # Using red-blue colormap
    cbar_a = fig_ab_diff.colorbar(a_plot, ax=axes_ab_diff[0], fraction=0.046, pad=0.04)
    axes_ab_diff[0].set_title("A Channel (Green-Red) Diff")
    cbar_a.set_label("Intensity")
    axes_ab_diff[0].axis("off")

    # Display B Channel (Blue-Yellow)
    b_plot = axes_ab_diff[1].imshow(img_skin_bch - dom_b, cmap="PuOr")  # Using purple-orange colormap
    cbar_b = fig_ab_diff.colorbar(b_plot, ax=axes_ab_diff[1], fraction=0.046, pad=0.04)
    axes_ab_diff[1].set_title("B Channel (Blue-Yellow) Diff")
    cbar_b.set_label("Intensity")
    axes_ab_diff[1].axis("off")

    plt.tight_layout()
    if(plt_show):
        plt.show()
    if(save_option):
        fig_ab_diff.savefig(output_path + '/' + image_file_name + '_LAB_diff.png', dpi=300, bbox_inches='tight')
    plt.close(fig_ab_diff)


def lab_diff_mask(image, mask, dom_a, dom_b, save_option=False, output_path=None, image_file_name=None, plt_show=False):

    img_skin_ach = image[:,:,1]
    img_skin_bch = image[:,:,2]   

    # Create a figure with two subplots for diff plot
    img_skin_ach_diff_mask = np.where(img_skin_ach - dom_a > 0, img_skin_ach - dom_a, 0)
    img_skin_bch_diff_mask = np.where(img_skin_bch - dom_b > 0, img_skin_bch - dom_b, 0)

    # Find min and max values for proper color scaling
    ach_min, ach_max = np.min(img_skin_ach_diff_mask), np.max(img_skin_ach_diff_mask)
    bch_min, bch_max = np.min(img_skin_bch_diff_mask), np.max(img_skin_bch_diff_mask)   
    
    fig_ab_diff_mask, axes_ab_diff_mask = plt.subplots(1, 2, figsize=(12, 6))

    # Display A Channel (Green-Red)
    a_plot = axes_ab_diff_mask[0].imshow(img_skin_ach_diff_mask, cmap="RdBu_r", vmin=ach_min, vmax=ach_max)  # Using red-blue colormap
    cbar_a = fig_ab_diff_mask.colorbar(a_plot, ax=axes_ab_diff_mask[0], fraction=0.046, pad=0.04)
    axes_ab_diff_mask[0].set_title("A Channel (Green-Red) Diff Mask")
    cbar_a.set_label("Intensity")
    axes_ab_diff_mask[0].axis("off")

    # Display B Channel (Blue-Yellow)
    b_plot = axes_ab_diff_mask[1].imshow(img_skin_bch_diff_mask, cmap="PuOr", vmin=bch_min, vmax=bch_max)  # Using purple-orange colormap
    cbar_b = fig_ab_diff_mask.colorbar(b_plot, ax=axes_ab_diff_mask[1], fraction=0.046, pad=0.04)
    axes_ab_diff_mask[1].set_title("B Channel (Blue-Yellow) Diff Mask")
    cbar_b.set_label("Intensity")
    axes_ab_diff_mask[1].axis("off")

    plt.tight_layout()
    if(plt_show):
        plt.show()
    if(save_option):
        fig_ab_diff_mask.savefig(output_path + '/' + image_file_name + '_LAB_diff_mask.png', dpi=300, bbox_inches='tight')
    plt.close(fig_ab_diff_mask)

    return img_skin_ach_diff_mask, img_skin_bch_diff_mask