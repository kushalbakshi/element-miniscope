import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours


def plot_all_rois(correlation_image, roi_contours, figsize=(10, 8), seed=42):
    """Optimized version that uses pre-computed contours"""
    np.random.seed(seed)
    colors = [np.maximum(np.random.rand(3), 0.3) for _ in range(len(roi_contours))]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(correlation_image, cmap="viridis", aspect="equal")
    ax.set_title("All ROI Contours on Correlation Image", fontsize=14, fontweight="bold")
    
    for color, contours in zip(colors, roi_contours):
        if contours is not None:
            for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlim(0, correlation_image.shape[1])
    ax.set_ylim(correlation_image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def plot_highlighted_roi(correlation_image, roi_contours, fluorescence_traces,
                                    roi_to_highlight, roi_id, figsize=(12, 8)):
    """Optimized version that uses pre-computed contours"""
    fig, (ax_spatial, ax_trace) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )
    
    # Spatial plot
    ax_spatial.imshow(correlation_image, cmap="viridis", aspect="equal")
    ax_spatial.set_title(f"ROI {roi_id} Highlighted", fontsize=14, fontweight="bold")
    
    for idx, contours in enumerate(roi_contours):
        if contours is not None:
            color = "red" if idx == roi_to_highlight else "gray"
            linewidth = 3 if idx == roi_to_highlight else 1.5
            alpha = 1.0 if idx == roi_to_highlight else 0.7
            
            for contour in contours:
                ax_spatial.plot(contour[:, 0], contour[:, 1], 
                                color=color, linewidth=linewidth, alpha=alpha)
    
    ax_spatial.set_xlim(0, correlation_image.shape[1])
    ax_spatial.set_ylim(correlation_image.shape[0], 0)
    ax_spatial.set_xticks([])
    ax_spatial.set_yticks([])
    
    # Fluorescence trace
    if roi_to_highlight < len(fluorescence_traces):
        trace_data = fluorescence_traces[roi_to_highlight]
        timepoints = np.arange(len(trace_data))
        ax_trace.plot(timepoints, trace_data, color="red", linewidth=2)
        ax_trace.set_xlabel("Frame #", fontsize=12)
        ax_trace.set_ylabel("Fluorescence", fontsize=12)
        ax_trace.set_title(f"ROI {roi_id} Fluorescence Trace", fontsize=14, fontweight="bold")
        ax_trace.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def get_roi_contours_from_pixels(roi_data, d1, d2, thr=0.99):
    """
    Convert ROI pixel data to contours (same logic as before)
    """

    x_pix = roi_data["mask_xpix"]
    y_pix = roi_data["mask_ypix"]

    # Get weights
    if roi_data["mask_weights"] is not None:
        weights = roi_data["mask_weights"]
    else:
        weights = np.ones(len(x_pix))

    # Create sparse representation
    total_pixels = d1 * d2
    A_component = np.zeros(total_pixels)

    # Convert 2D coordinates to linear indices
    linear_indices = y_pix + x_pix * d1

    # Handle potential out-of-bounds indices
    valid_mask = (linear_indices >= 0) & (linear_indices < total_pixels)
    linear_indices = linear_indices[valid_mask]
    weights = weights[valid_mask]

    if len(linear_indices) == 0:
        return None

    # Set weights in the component vector
    A_component[linear_indices] = weights

    # Apply energy thresholding
    patch_data = A_component[A_component > 0]
    if len(patch_data) == 0:
        return None

    indx = np.argsort(patch_data)[::-1]
    cumEn = np.cumsum(patch_data[indx] ** 2)
    cumEn /= cumEn[-1]

    # Create thresholded version
    Bvec = np.ones(len(A_component))
    indices = np.where(A_component > 0)[0]
    Bvec[indices[indx]] = cumEn
    Bmat = Bvec.reshape((d1, d2), order="F")

    # Find contours
    try:
        vertices = find_contours(Bmat.T, thr)
        if len(vertices) == 0:
            return None
        return vertices
    except:
        return None
