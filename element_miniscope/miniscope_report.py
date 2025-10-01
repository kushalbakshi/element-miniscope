import datajoint as dj
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from . import miniscope

schema = dj.schema()


def activate(schema_name, *, create_schema=True, create_tables=True):
    """Activate this schema.

    The "activation" of miniscope_report should be evoked by the miniscope module

    Args:
        schema_name (str): schema name on the database server to activate the
            `miniscope_report` schema
        create_schema (bool): when True (default), create schema in the database if it
            does not yet exist.
        create_tables (str): when True (default), create schema takes in the database
            if they do not yet exist.
    """
    if not miniscope.schema.is_activated():
        raise RuntimeError("Please activate the `miniscope` schema first.")

    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=miniscope.__dict__,
    )


@schema
class QualityMetrics(dj.Imported):
    definition = """
    -> miniscope.Processing
    ---
    r_values=null  : longblob # space correlation for each component
    snr=null       : longblob # trace SNR for each component
    cnn_preds=null : longblob # CNN predictions for each component
    """

    def make(self, key):
        from .miniscope import get_loader_result

        method, loaded_result = get_loader_result(key, miniscope.Curation)
        assert (
            method == "caiman"
        ), f"Quality figures for {method} not yet implemented. Try CaImAn."

        key.update(
            {
                attrib_name: getattr(loaded_result.cnmf.estimates, attrib, None)
                for attrib_name, attrib in zip(
                    ["r_values", "snr", "cnn_preds"],
                    ["r_values", "SNR_comp", "cnn_preds"],
                )
            }
        )

        self.insert1(key)


@schema
class MiniscopeMaskOverlayPlots(dj.Computed):
    definition = """
    -> miniscope.Fluorescence
    ---
    correlation_image_overlay_png: attach  # ROIs overlayed on correlation image
    """

    class CorrelationImageByRoi(dj.Part):
        definition = """
        -> master
        -> miniscope.Fluorescence.Trace
        ---
        image_by_roi_png: attach  # ROIs overlayed on correlation image
        """

    def make_fetch(self, key):
        image_dims = (miniscope.RecordingInfo & key).fetch1("px_height", "px_width")
        corr_img = (miniscope.MotionCorrection.Summary & key).fetch1("correlation_image")
        roi_data = (miniscope.Segmentation.Mask & key).fetch(
            "mask", "mask_xpix", "mask_ypix", "mask_weights", 
            as_dict=True, order_by="mask ASC"
        )
        fluorescence_traces = (miniscope.Fluorescence.Trace & key & "fluorescence_channel=0").fetch(
            "fluorescence", order_by="mask ASC"
        )
        return image_dims, corr_img, roi_data, fluorescence_traces

    def make_compute(self, key, image_dims, corr_img, roi_data, fluorescence_traces):
        from .plotting.cell_plot import plot_all_rois, plot_highlighted_roi, get_roi_contours_from_pixels
        
        
        tmpdir = tempfile.TemporaryDirectory()
        if len(corr_img.shape) == 3:
            corr_img = corr_img[0, :, :]  # Use first channel
        
        d1, d2 = image_dims
        roi_contours = []
        for roi in roi_data:
            contours = get_roi_contours_from_pixels(roi, d1, d2, thr=0.99)
            roi_contours.append(contours)
        
        # Create all-ROI overlay plot
        image_overlay_fig = plot_all_rois(
            corr_img, roi_contours
        )
        correlation_image_overlay_file = Path(tmpdir.name) / "correlation_image_overlay.png"
        image_overlay_fig.savefig(correlation_image_overlay_file, format="png", bbox_inches="tight", dpi=100)
        plt.close(image_overlay_fig)
        
        # Create individual ROI plots
        image_by_roi_overlays_files = []
        part_inserts = []
        
        for idx, roi in enumerate(roi_data):
            mask_id = roi["mask"]
            fig = plot_highlighted_roi(
                corr_img, roi_contours, fluorescence_traces, 
                roi_to_highlight=idx, roi_id=mask_id
            )
            
            filepath = Path(tmpdir.name) / f"image_by_roi_{mask_id}.png"
            fig.savefig(filepath, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            
            image_by_roi_overlays_files.append(filepath)
            part_inserts.append({**key, "fluorescence_channel": 0, "mask": mask_id, "image_by_roi_png": filepath})
        
        return correlation_image_overlay_file, part_inserts, tmpdir
    
    def make_insert(self, key, correlation_image_overlay_file, part_inserts, tmpdir):
        self.insert1({**key, "correlation_image_overlay_png": correlation_image_overlay_file})
        self.CorrelationImageByRoi.insert(part_inserts)
        tmpdir.cleanup()