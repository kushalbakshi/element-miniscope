import csv
import copy
import cv2
import gc
import importlib
import inspect
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import (
    dict_to_uuid,
    find_full_path,
    find_root_directory,
    memoized_result,
)

logger = dj.logger

schema = dj.schema()

_linking_module = None


def activate(
    miniscope_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        miniscope_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
            does not yet exist.
        create_tables (str): when True (default), create schema takes in the database
            if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:

    Upstream tables:
        Session: parent table to Recording, identifying a recording session.
        Device: Reference table for Recording, specifying the acquisition device.
        AnatomicalLocation: Reference table for anatomical region for recording acquisition.

    Functions:
        get_miniscope_root_data_dir(): Returns absolute path for root data director(y/ies)
            with all subject/sessions data, as (list of) string(s).
        get_session_directory(session_key: dict) Returns the session directory with all
            data for the session in session_key, as a string.
        get_processed_root_data_dir(): Returns absolute path for all processed data as
            a string.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    schema.activate(
        miniscope_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# Functions required by the element-miniscope  -----------------------------------------


def get_miniscope_root_data_dir() -> list:
    """Fetches absolute data path to miniscope data directory.

    The absolute path here is used as a reference for all downstream relative paths used in DataJoint.

    Returns:
        A list of the absolute path to miniscope data directory.
    """

    root_directories = _linking_module.get_miniscope_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_processed_root_data_dir() -> Union[str, pathlib.Path]:
    """Retrieve the root directory for all processed data.

    All data paths and directories in DataJoint Elements are recommended to be stored as
    relative paths (posix format), with respect to some user-configured "root"
    directory, which varies from machine to machine (e.g. different mounted drive
    locations).

    Returns:
        dir (str| pathlib.Path): Absolute path of the processed miniscope root data
            directory.
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_miniscope_root_data_dir()[0]


def get_session_directory(session_key: dict) -> str:
    """Pulls session directory information from database.

    Args:
        session_key (dict): a dictionary containing session information.

    Returns:
        Session directory as a string.
    """
    return _linking_module.get_session_directory(session_key)


# Experiment and analysis meta information -------------------------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    """Software used for miniscope acquisition.

    Required to define a miniscope recording.

    Attributes:
        acq_software (str): Name of the miniscope acquisition software."""

    definition = """
    acq_software: varchar(24)
    """
    contents = zip(["Miniscope-DAQ-V3", "Miniscope-DAQ-V4", "Inscopix", "Bonsai"])


@schema
class Channel(dj.Lookup):
    """Number of channels in the miniscope recording.

    Attributes:
        channel (tinyint): Number of channels in the miniscope acquisition starting at zero.
    """

    definition = """
    channel     : tinyint  # 0-based indexing
    """
    contents = zip(range(5))


@schema
class Recording(dj.Manual):
    """Recording defined by a measurement done using a scanner and an acquisition software.

    Attributes:
        Session (foreign key): A primary key from Session.
        recording_id (int): Unique recording ID.
        Device (foreign key, optional): A primary key from Device.
        AcquisitionSoftware (foreign key): A primary key from AcquisitionSoftware.
        recording_notes (str, optional): notes about the recording session.
    """

    definition = """
    -> Session
    recording_id: int
    ---
    -> [nullable] Device
    -> AcquisitionSoftware
    recording_notes='' : varchar(4095) # free-notes
    """


@schema
class RecordingLocation(dj.Manual):
    """Brain location where the miniscope recording is acquired.

    Attributes:
        Recording (foreign key): A primary key from Recording.
        AnatomicalLocation (foreign key): A primary key from AnatomicalLocation.
    """

    definition = """
    # Brain location where this miniscope recording is acquired
    -> Recording
    ---
    -> AnatomicalLocation
    """


@schema
class RecordingInfo(dj.Imported):
    """Information about the recording extracted from the recorded files.

    Attributes:
        Recording (foreign key): A primary key from Recording.
        nchannels (tinyint): Number of recording channels.
        nframes (int): Number of recorded frames.
        px_height (smallint): Height in pixels.
        px_width (smallint): Width in pixels.
        um_height (float): Height in microns.
        um_width (float): Width in microns.
        fps (float): Frames per second, (Hz).
        gain (float): Recording gain.
        spatial_downsample (tinyint): Amount of downsampling applied.
        led_power (float): LED power used for the recording.
        recording_datetime (datetime): Datetime of the recording.
        recording_duration (float): Total recording duration (seconds).
    """

    definition = """
    # Store metadata about recording
    -> Recording
    ---
    nchannels            : tinyint   # number of channels
    nframes              : int       # number of recorded frames
    ndepths=1            : tinyint   # number of depths
    px_height            : smallint  # height in pixels
    px_width             : smallint  # width in pixels
    fps                  : float     # (Hz) frames per second
    recording_datetime=null   : datetime  # datetime of the recording
    recording_duration=null   : float     # (seconds) duration of the recording
    """

    class Config(dj.Part):
        """Recording metadata and configuration.

        Attributes:
            Recording (foreign key): A primary key from RecordingInfo.
            config (longblob): Recording metadata and configuration.
        """

        definition = """
        -> master
        ---
        config: longblob  # recording metadata and configuration
        """

    class Timestamps(dj.Part):
        """Recording timestamps for each frame.

        Attributes:
            Recording (foreign key): A primary key from RecordingInfo.
            timestamps (longblob): Recording timestamps for each frame.
        """

        definition = """
        -> master
        ---
        timestamps: longblob
        """

    class File(dj.Part):
        """File path to recording file relative to root data directory.

        Attributes:
            Recording (foreign key): Recording primary key.
            file_id (foreign key, smallint): Unique file ID.
            path_path (varchar(255) ): Relative file path to recording file.
        """

        definition = """
        -> master
        file_id : smallint unsigned
        ---
        file_path: varchar(255)      # relative to root data directory
        """

    def make(self, key):
        """Populate table with recording file metadata."""

        # Search recording directory for miniscope raw files
        acq_software = (Recording & key).fetch1("acq_software")
        recording_directory = get_session_directory(key)

        recording_path = find_full_path(
            get_miniscope_root_data_dir(), recording_directory
        )

        recording_filepaths = (
            [file_path.as_posix() for file_path in recording_path.glob("*.avi")]
            if acq_software != "Inscopix"
            else [file_path.as_posix() for file_path in recording_path.rglob("*.avi")]
        )
        if not recording_filepaths:
            raise FileNotFoundError(f"No .avi files found in " f"{recording_directory}")

        if acq_software == "Miniscope-DAQ-V3":
            recording_timestamps = recording_path / "timestamp.dat"
            if not recording_timestamps.exists():
                raise FileNotFoundError(
                    f"No timestamp file found in " f"{recording_directory}"
                )

            nchannels = 1  # Assumes a single channel

            # Parse number of frames from timestamp.dat file
            with open(recording_timestamps) as f:
                next(f)
                nframes = sum(1 for line in f if int(line[0]) == 0)

            # Parse image dimension and frame rate
            video = cv2.VideoCapture(recording_filepaths[0])
            _, frame = video.read()
            frame_size = np.shape(frame)
            px_height = frame_size[0]
            px_width = frame_size[1]

            fps = video.get(cv2.CAP_PROP_FPS)

        elif acq_software == "Miniscope-DAQ-V4":
            metadata = None  # Initialize to handle the no-metadata case

            try:
                recording_metadata = next(recording_path.glob("metaData.json"))
                with open(recording_metadata.as_posix()) as f:
                    metadata = json.loads(f.read())
                px_height = metadata["ROI"]["height"]
                px_width = metadata["ROI"]["width"]
                fps = int(metadata["frameRate"].replace("FPS", ""))
            except StopIteration:
                logger.warning(
                    f"No metaData.json file found in {recording_directory}\n"
                    "Extracting metadata from the .avi file header instead."
                )
                miniscope_video = cv2.VideoCapture(recording_filepaths[0])
                px_height = int(miniscope_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                px_width = int(miniscope_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                fps = miniscope_video.get(cv2.CAP_PROP_FPS)
                miniscope_video.release()

            # Handle timestamps for multiple vs single AVI files
            time_stamps = None
            if len(recording_filepaths) > 1:
                # Multiple AVI files - look for timestamp CSV paired with each AVI
                timestamps_files = sorted(recording_path.glob("*.csv"))
                all_timestamps = []
                for timestamps_file in timestamps_files:
                    with open(timestamps_file, newline="") as f:
                        reader = csv.reader(f, delimiter=",")
                        next(reader)  # Skip header for each file
                        all_timestamps.extend(list(reader))
                if all_timestamps:
                    time_stamps = np.array(all_timestamps, dtype=float)[:, 0]
                    nframes = len(time_stamps)
                else:
                    logger.warning(
                        f"No timestamp CSV files found in {recording_directory}"
                    )
            else:
                try:
                    recording_timestamps = next(recording_path.glob("timeStamps.csv"))
                    with open(recording_timestamps, newline="") as f:
                        reader = csv.reader(f, delimiter=",")
                        next(reader)  # Skip header
                        time_stamps = np.array(list(reader), dtype=float)[:, 0]
                    nframes = len(time_stamps)
                except StopIteration:
                    logger.warning(
                        f"No timeStamps.csv file found in {recording_directory}"
                    )

            # Fallback: get nframes from video if timestamps not available
            if time_stamps is None or len(time_stamps) == 0:
                total_frames = 0
                for avi_file in recording_filepaths:
                    video = cv2.VideoCapture(avi_file)
                    total_frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                nframes = total_frames

            nchannels = 1  # Assumes a single channel

        elif acq_software == "Inscopix":
            inscopix_metadata = next(recording_path.glob("session.json"))
            timestamps_file = next(recording_path.glob("*/*timestamps.csv"))
            metadata = json.load(open(inscopix_metadata))
            recording_timestamps = pd.read_csv(timestamps_file)

            nchannels = len(metadata["manual"]["mScope"]["ledMaxPower"])
            nframes = len(recording_timestamps)
            fps = metadata["microscope"]["fps"]["fps"]
            time_stamps = (recording_timestamps[" time (ms)"] / 1000).values
            px_height = metadata["microscope"]["fov"]["height"]
            px_width = metadata["microscope"]["fov"]["width"]

        elif acq_software == "Bonsai":
            logger.warning(
                f"Limited support for Bonsai recordings. Metadata will be extracted directly from the `.avi` files. To improve support, please contact the developers or open an issue on the GitHub repository."
            )

            miniscope_video = cv2.VideoCapture(recording_filepaths[0])

            nchannels = 1
            nframes = int(miniscope_video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = miniscope_video.get(cv2.CAP_PROP_FPS)
            px_height = int(miniscope_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            px_width = int(miniscope_video.get(cv2.CAP_PROP_FRAME_WIDTH))

        else:
            raise NotImplementedError(
                f"Loading routine not implemented for {acq_software}"
                " acquisition software"
            )

        # Insert in RecordingInfo
        self.insert1(
            dict(
                key,
                nchannels=nchannels,
                nframes=nframes,
                px_height=px_height,
                px_width=px_width,
                fps=fps,
                recording_duration=nframes / fps,
            )
        )

        # Insert file(s)
        recording_files = [
            pathlib.Path(f)
            .relative_to(find_root_directory(get_miniscope_root_data_dir(), f))
            .as_posix()
            for f in recording_filepaths
        ]

        self.File.insert(
            [
                {**key, "file_id": i, "file_path": f}
                for i, f in enumerate(recording_files)
            ]
        )

        if acq_software == "Inscopix":
            self.Timestamps.insert1(dict(**key, timestamps=time_stamps))
            self.Config.insert1(dict(**key, config=metadata))
        elif acq_software == "Miniscope-DAQ-V4":
            if time_stamps is not None and len(time_stamps) > 0:
                self.Timestamps.insert1(dict(**key, timestamps=time_stamps))
            if metadata is not None:
                self.Config.insert1(dict(**key, config=metadata))


# Trigger a processing routine -------------------------------------------------


@schema
class ProcessingMethod(dj.Lookup):
    """Package used for processing of miniscope data (e.g. CaImAn, etc.).

    Attributes:
        processing_method (str): Processing method.
        processing_method_desc (str): Processing method description.
    """

    definition = """# Package used for processing of calcium imaging data (e.g. Suite2p, CaImAn, etc.).
    processing_method: varchar(16)
    ---
    processing_method_desc: varchar(1000)
    """

    contents = [
        ("caiman", "caiman analysis suite"),
        ("minian", "minian analysis suite"),
    ]


@schema
class ProcessingParamSet(dj.Lookup):
    """Parameter set used for the processing of miniscope recordings.,
    including both the analysis suite and its respective input parameters.

    A hash of the parameters of the analysis suite is also stored in order
    to avoid duplicated entries.

    Attributes:
        paramset_idx (int): Unique parameter set ID.
        ProcessingMethod (foreign key): A primary key from ProcessingMethod.
        paramset_desc (str): Parameter set description.
        paramset_set_hash (uuid): A universally unique identifier for the parameter set.
        params (longblob): Parameter set, a dictionary of all applicable parameters to the analysis suite.
    """

    definition = """# Processing Parameter set
    paramset_idx:  smallint     # Unique parameter set ID.
    ---
    -> ProcessingMethod
    paramset_desc: varchar(1280)    # Parameter set description
    param_set_hash: uuid    # A universally unique identifier for the parameter set unique index (param_set_hash)
    params: longblob  # Parameter set, a dictionary of all applicable parameters to the analysis suite.
    """

    @classmethod
    def insert_new_params(
        cls,
        processing_method: str,
        paramset_idx: int,
        paramset_desc: str,
        params: dict,
    ):
        """Insert new parameter set.

        Args:
            processing_method (str): Name of the processing method or software.
            paramset_idx (int): Unique number for the set of processing parameters.
            paramset_desc (str): Description of the processing parameter set.
            params (dict): Dictionary of processing parameters for the selected processing_method.
            processing_method_desc (str, optional): Description of the processing method. Defaults to "".

        Raises:
            dj.DataJointError: A parameter set with arguments in this function already exists in the database.
        """

        ProcessingMethod.insert1(
            {
                "processing_method": processing_method,
                "processing_method_desc": "caiman_analysis",
            },
            skip_duplicates=True,
        )
        param_dict = {
            "processing_method": processing_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        q_param = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1("paramset_idx")
            if pname == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, try adding with different name
                raise dj.DataJointError(
                    "The specified param-set already exists - name: {}".format(pname)
                )
        else:
            cls.insert1(param_dict)


@schema
class MaskType(dj.Lookup):
    """Possible classifications of a segmented mask.

    Attributes:
        mask_type (foreign key, varchar(16) ): Type of segmented mask.
    """

    definition = """ # Possible classifications for a segmented mask
    mask_type        : varchar(16)
    """

    contents = zip(["soma", "axon", "dendrite", "neuropil", "artefact", "unknown"])


@schema
class ProcessingTask(dj.Manual):
    """A pairing of processing params and recordings to be loaded or triggered.

    This table defines a miniscope recording processing task for a combination of a
    `Recording` and a `ProcessingParamSet` entries, including all the inputs (recording, method,
    method's parameters). The task defined here is then run in the downstream table
    `Processing`. This table supports definitions of both loading of pre-generated results
    and the triggering of new analysis for all supported analysis methods.

    Attributes:
        Recording (foreign key): Primary key from Recording.
        ProcessingParamSet (foreign key): Primary key from ProcessingParamSet.
        processing_output_dir (str): Output directory of the processed scan relative to the root data directory.
        task_mode (str): One of 'load' (load computed analysis results) or 'trigger'
            (trigger computation).
    """

    definition = """# Manual table for defining a processing task ready to be run
    -> Recording
    -> ProcessingParamSet
    ---
    processing_output_dir='': varchar(255)    # relative to the root data directory
    task_mode='load'      : enum('load', 'trigger') # 'load': load existing results
                                                    # 'trigger': trigger procedure
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """Infer an output directory for an entry in ProcessingTask table.

        Args:
            key (dict): Primary key from the ProcessingTask table.
            relative (bool): If True, processing_output_dir is returned relative to
                imaging_root_dir. Default False.
            mkdir (bool): If True, create the processing_output_dir directory.
                Default True.

        Returns:
            dir (str): A default output directory for the processed results (processed_output_dir
                in ProcessingTask) based on the following convention:
                processed_dir / scan_dir / {processing_method}_{paramset_idx}
                e.g.: sub4/sess1/scan0/suite2p_0
        """
        acq_software = (Recording & key).fetch1("acq_software")
        recording_dir = find_full_path(
            get_miniscope_root_data_dir(),
            get_session_directory(key),
        )
        root_dir = find_root_directory(get_miniscope_root_data_dir(), recording_dir)

        method = (
            (ProcessingParamSet & key).fetch1("processing_method").replace(".", "-")
        )

        processed_dir = pathlib.Path(get_processed_root_data_dir())
        output_dir = (
            processed_dir
            / recording_dir.relative_to(root_dir)
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def generate(cls, recording_key, paramset_idx=0):
        """Generate a ProcessingTask for a Recording using an parameter ProcessingParamSet

        Generate an entry in the ProcessingTask table for a particular recording using an
        existing parameter set from the ProcessingParamSet table.

        Args:
            recording_key (dict): Primary key from Recording.
            paramset_idx (int): Unique parameter set ID.
        """
        key = {**recording_key, "paramset_idx": paramset_idx}

        processed_dir = get_processed_root_data_dir()
        output_dir = cls.infer_output_dir(key, relative=False, mkdir=True)

        method = (ProcessingParamSet & {"paramset_idx": paramset_idx}).fetch1(
            "processing_method"
        )

        try:
            if method == "caiman":
                from element_interface import caiman_loader

                caiman_loader.CaImAn(output_dir)
            else:
                raise NotImplementedError(
                    "Unknown/unimplemented method: {}".format(method)
                )
        except FileNotFoundError:
            task_mode = "trigger"
        else:
            task_mode = "load"

        cls.insert1(
            {
                **key,
                "processing_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
                "task_mode": task_mode,
            }
        )

    auto_generate_entries = generate


@schema
class Processing(dj.Computed):
    """Perform the computation of an entry (task) defined in the ProcessingTask table.
    The computation is performed only on the recordings with RecordingInfo inserted.


    Attributes:
        ProcessingTask (foreign key): Primary key from ProcessingTask.
        processing_time (datetime): Process completion datetime.
        package_version (str, optional): Version of the analysis package used in processing the data.
    """

    definition = """
    -> ProcessingTask
    ---
    processing_time     : datetime  # generation time of processed results
    package_version=''  : varchar(16)
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)     # file name
        ---
        file: filepath@miniscope-processed
        """

    @property
    def key_source(self):
        return ProcessingTask & RecordingInfo

    def make_fetch(self, key):
        task_mode, processing_output_dir = (ProcessingTask & key).fetch1(
            "task_mode", "processing_output_dir"
        )
        method = (ProcessingParamSet & key).fetch1("processing_method")
        avi_files = (RecordingInfo.File & key).fetch("file_path")
        processing_params = (ProcessingParamSet & key).fetch1("params")
        sampling_rate = (RecordingInfo & key).fetch1("fps")
        px_height, px_width = (RecordingInfo & key).fetch1("px_height", "px_width")

        return (
            task_mode,
            processing_output_dir,
            method,
            avi_files,
            processing_params,
            sampling_rate,
            px_height,
            px_width,
        )

    def make_compute(
        self,
        key,
        task_mode,
        processing_output_dir,
        method,
        avi_files,
        processing_params,
        sampling_rate,
        px_height,
        px_width,
    ):
        """
        Execute the miniscope analysis defined by the ProcessingTask.
        - task_mode: 'load', confirm that the results are already computed.
        - task_mode: 'trigger' runs the analysis.
        """
        if method not in ["caiman", "minian"]:
            raise NotImplementedError(f"Method {method} is not supported")

        params = copy.deepcopy(processing_params)

        if not processing_output_dir:
            output_dir = ProcessingTask.infer_output_dir(key, relative=True, mkdir=True)
        else:
            output_dir = processing_output_dir
        try:
            output_dir = find_full_path(get_processed_root_data_dir(), output_dir)
        except FileNotFoundError as e:
            if task_mode == "trigger":
                processed_dir = pathlib.Path(get_processed_root_data_dir())
                output_dir = processed_dir / output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise e

        if task_mode == "load":
            method, loaded_result = get_loader_result(key, ProcessingTask)
            if method == "caiman":
                loaded_caiman = loaded_result
                key = {**key, "processing_time": loaded_caiman.creation_time}
            elif method == "minian":
                loaded_minian = loaded_result
                key = {**key, "processing_time": loaded_minian.creation_time}
            else:
                raise NotImplementedError(
                    f"Loading of {method} data is not yet supported"
                )
        elif task_mode == "trigger":
            avi_files = [
                find_full_path(get_miniscope_root_data_dir(), avi_file).as_posix()
                for avi_file in avi_files
            ]
            if method == "caiman":
                import multiprocessing
                import caiman as cm
                from caiman.motion_correction import MotionCorrect
                from caiman.source_extraction.cnmf.cnmf import CNMF
                from caiman.source_extraction.cnmf.params import CNMFParams
                from element_interface.run_caiman import _save_mc

                extra_params = params.pop("extra_dj_params", {})

                params["fnames"] = avi_files
                params["fr"] = sampling_rate
                params["is3D"] = False
                if "indices" in params:
                    params["motion"] = {
                        "indices": (
                            slice(*params.get("indices")[0]),
                            slice(*params.get("indices")[1]),
                        )
                    }
                else:
                    params["motion"] = {"indices": (slice(None), slice(None))}

                @memoized_result(
                    uniqueness_dict=params,
                    output_directory=output_dir,
                )
                def _run_processing():
                    mc_indices = params["motion"].get("indices")
                    caiman_temp = os.environ.get("CAIMAN_TEMP")
                    os.environ["CAIMAN_TEMP"] = str(output_dir)
                    n_processes = np.floor(multiprocessing.cpu_count() * 0.6)
                    n_processes = int(os.getenv("CAIMAN_MC_N_PROCESSES", n_processes))
                    _, dview, n_processes = cm.cluster.setup_cluster(
                        backend="multiprocessing",
                        n_processes=n_processes,
                        maxtasksperchild=1,
                    )
                    try:
                        opts = CNMFParams(params_dict=params)
                        cnm = CNMF(n_processes, params=opts, dview=dview)
                        fnames = cnm.params.get("data", "fnames")
                        mc = MotionCorrect(fnames, dview=cnm.dview, **cnm.params.motion)
                        mc_base_attrs = list(mc.__dict__)
                        logger.info("Starting motion correction (CaImAn)...")
                        mc.motion_correct(save_movie=mc_indices is None)
                        mc_results = {
                            k: v
                            for k, v in mc.__dict__.items()
                            if k not in mc_base_attrs
                        }
                        if cnm.params.get("motion", "pw_rigid"):
                            mc_results["b0"] = np.ceil(
                                np.max(np.abs(mc.shifts_rig))
                            ).astype(int)
                            cnm.estimates.shifts = mc.shifts_rig
                            if cnm.params.get("motion", "is3D"):
                                cnm.estimates.shifts = [
                                    mc.x_shifts_els,
                                    mc.y_shifts_els,
                                    mc.z_shifts_els,
                                ]
                            else:
                                cnm.estimates.shifts = [
                                    mc.x_shifts_els,
                                    mc.y_shifts_els,
                                ]
                        else:
                            mc_results["b0"] = np.ceil(
                                np.max(np.abs(mc.shifts_rig))
                            ).astype(int)
                            cnm.estimates.shifts = mc.shifts_rig

                        base_name = pathlib.Path(fnames[0]).stem
                        fname_mc = (
                            mc.fname_tot_els
                            if cnm.params.motion["pw_rigid"]
                            else mc.fname_tot_rig
                        )
                        if all(fname_mc):
                            logger.info("Generating C-order memmap file...")
                            border_to_0 = (
                                0 if mc.border_nan == "copy" else mc.border_to_0
                            )
                            fname_new = cm.mmapping.save_memmap(
                                fname_mc,
                                base_name=base_name + "_mc",
                                order="C",
                                var_name_hdf5=cnm.params.get("data", "var_name_hdf5"),
                                border_to_0=border_to_0,
                            )
                        else:
                            logger.info(
                                "Applying shifts, then generating C-order memmap file..."
                            )
                            fname_new = mc.apply_shifts_movie(
                                fnames,
                                save_memmap=True,
                                save_base_name=base_name + "_mc",
                                order="C",
                            )
                            mc.mmap_file = [fname_new]
                        Yr, dims, T = cm.mmapping.load_memmap(fname_new)
                        images = np.reshape(Yr.T, [T] + list(dims), order="F")
                        cnm.mmap_file = fname_new
                        # terminate the previous cluster and setup a new one with fewer
                        # processes for CNMF because it is memory intensive
                        dview.terminate()
                        n_processes = np.floor(multiprocessing.cpu_count() * 0.2)
                        n_processes = int(
                            os.getenv("CAIMAN_CNMF_N_PROCESSES", n_processes)
                        )
                        _, dview, n_processes = cm.cluster.setup_cluster(
                            backend="multiprocessing",
                            n_processes=n_processes,
                            maxtasksperchild=1,
                        )
                        cnm.dview = dview
                        logger.info(
                            f"Starting CNMF analysis with {n_processes} processes..."
                        )

                        cnm.fit(images, indices=(slice(None), slice(None)))
                        cnm.estimates.evaluate_components(
                            images, cnm.params, dview=cnm.dview
                        )
                        cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)
                        logger.info("Computing summary images...")
                        correlation_image, _ = cm.summary_images.correlation_pnr(
                            images[:: max(T // 1000, 1)],
                            gSig=cnm.params.init["gSig"][0],
                            swap_dim=False,
                        )
                        correlation_image[np.isnan(correlation_image)] = 0
                        cnm.estimates.Cn = correlation_image
                        fname_hdf5 = cnm.mmap_file[:-4] + "hdf5"
                        cnm.save(fname_hdf5)
                        cnmf_output_file = pathlib.Path(fname_hdf5)
                        summary_images = {
                            "average_image": np.mean(
                                images[:: max(T // 1000, 1)], axis=0
                            ),
                            "max_image": np.max(images[:: max(T // 1000, 1)], axis=0),
                            "correlation_image": correlation_image,
                        }
                        _save_mc(
                            mc,
                            cnmf_output_file.as_posix(),
                            params["is3D"],
                            summary_images=summary_images,
                        )
                    except Exception as e:
                        dview.terminate()
                        raise e
                    else:
                        cm.stop_server(dview=dview)
                        logger.info("CNMF analysis complete. Resulted saved.")
                        caiman_temp = os.environ.get("CAIMAN_TEMP")
                        if caiman_temp is not None:
                            os.environ["CAIMAN_TEMP"] = caiman_temp
                        else:
                            del os.environ["CAIMAN_TEMP"]

                _run_processing()
                _, imaging_dataset = get_loader_result(
                    key, ProcessingTask, full_output_dir=output_dir
                )
                caiman_dataset = imaging_dataset
                key["processing_time"] = caiman_dataset.creation_time
                key["package_version"] = cm.__version__
                file_entries = [
                    {
                        **key,
                        "file_name": f.relative_to(
                            get_processed_root_data_dir()
                        ).as_posix(),
                        "file": f.as_posix(),
                    }
                    for f in output_dir.rglob("*")
                    if f.is_file()
                ]

            elif method == "minian":
                import multiprocessing
                import psutil
                import shutil
                import dask
                import dask.array as darr
                import scipy.linalg                          # <<< NEW: for solve_triangular patch
                import xarray as xr                          # <<< NEW: for sanitize_array
                from dask.distributed import Client, LocalCluster

                # <<< NEW: Prevent infinite retry loops on NaN/Inf errors >>>
                # If a Dask task fails (e.g. solve_triangular ValueError), don't
                # keep retrying until OOM — fail fast after 3 attempts.
                dask.config.set({
                    "distributed.scheduler.allowed-failures": 3,
                    "distributed.comm.timeouts.connect": "300s",
                    "distributed.comm.timeouts.tcp": "7200s",      # 2 hours
                    "distributed.worker.lifetime.stale": "7200s",   # don't mark workers stale quickly
                    "distributed.scheduler.work-stealing": False,
                })

                # ===== APPLY COMPATIBILITY PATCHES =====
                
                # Fix for NetworkX 3.0+ API changes
                import networkx as nx
                import scipy.sparse
                from scipy.sparse import issparse
                import minian.cnmf as cnmf_module

                def label_connected_fixed(adj, only_connected=False):
                    """Fixed label_connected for NetworkX 3.0+ compatibility."""
                    if issparse(adj):
                        adj = adj.toarray()
                    adj = adj.copy()
                    np.fill_diagonal(adj, 0)
                    adj = np.triu(adj)
                    g = nx.from_numpy_array(adj)
                    labels = np.zeros(adj.shape[0], dtype=int)
                    for icomp, comp in enumerate(nx.connected_components(g)):
                        for node in comp:
                            labels[node] = icomp
                    if only_connected:
                        iso_mask = np.array([len(c) == 1 for c in nx.connected_components(g)])
                        labels[np.isin(labels, np.where(iso_mask)[0])] = -1
                    return labels

                cnmf_module.label_connected = label_connected_fixed

                # Fix for sparse array auto-densification
                import sparse
                import sparse.numba_backend._sparse_array as sparse_mod
                sparse_mod.AUTO_DENSIFY = True

                # Fix for darr.block with mixed sparse array types
                _original_darr_block = darr.block

                def patched_darr_block(arrays, allow_unknown_chunksizes=False):
                    """Patched darr.block that ensures sparse array type consistency."""
                    def convert_to_coo(arr):
                        if arr is None:
                            return arr
                        if isinstance(arr, sparse.COO):
                            return arr
                        if isinstance(arr, sparse.SparseArray):
                            return sparse.COO(arr)
                        if isinstance(arr, np.ndarray):
                            return sparse.COO.from_numpy(arr)
                        if hasattr(arr, 'todense'):
                            return sparse.COO.from_numpy(np.asarray(arr.todense()))
                        return arr

                    def recursive_convert(obj):
                        if isinstance(obj, list):
                            return [recursive_convert(item) for item in obj]
                        elif isinstance(obj, np.ndarray) and obj.dtype == object:
                            result = np.empty_like(obj)
                            for idx in np.ndindex(obj.shape):
                                result[idx] = convert_to_coo(obj[idx])
                            return result
                        else:
                            return convert_to_coo(obj)

                    try:
                        return _original_darr_block(arrays, allow_unknown_chunksizes=allow_unknown_chunksizes)
                    except ValueError as e:
                        if "All arrays must be instances of SparseArray" in str(e):
                            converted = recursive_convert(arrays)
                            return _original_darr_block(converted, allow_unknown_chunksizes=allow_unknown_chunksizes)
                        raise

                darr.block = patched_darr_block
                darr.core.block = patched_darr_block

                # ===== PATCH: sparse/dense concatenation compatibility =====
                import sparse
                import sparse.numba_backend._coo.common as _sparse_coo_common
                import numpy as np

                _original_sparse_concat = _sparse_coo_common.concatenate

                def _patched_sparse_concat(arrays, axis=0):
                    """Convert any dense arrays to sparse.COO before concatenation."""
                    converted = []
                    for arr in arrays:
                        if isinstance(arr, sparse.SparseArray):
                            converted.append(arr)
                        elif isinstance(arr, np.ndarray):
                            converted.append(sparse.COO.from_numpy(arr))
                        else:
                            try:
                                converted.append(sparse.COO.from_numpy(np.asarray(arr)))
                            except Exception:
                                converted.append(arr)
                    return _original_sparse_concat(converted, axis=axis)

                _sparse_coo_common.concatenate = _patched_sparse_concat

                # ===== PATCH: sparse/dense concatenation compatibility =====
                import sparse
                import numpy as np
                import sparse.numba_backend._coo.common as _sparse_coo_common

                _original_check = _sparse_coo_common.check_consistent_fill_value

                def _patched_check(arrays):
                    """Auto-convert any dense arrays to sparse.COO before validation."""
                    for i, arr in enumerate(arrays):
                        if not isinstance(arr, sparse.SparseArray):
                            arrays[i] = sparse.COO.from_numpy(np.asarray(arr))
                    return _original_check(arrays)

                _sparse_coo_common.check_consistent_fill_value = _patched_check
                # Patch update_temporal to handle sparse.COO arrays
                _original_update_temporal = cnmf_module.update_temporal

                def patched_update_temporal(A, C, b=None, f=None, Y=None, YrA=None, 
                                            noise_freq=0.25, p=2, add_lag="p", jac_thres=0.1, 
                                            sparse_penal=1, bseg=None, med_wd=None, 
                                            zero_thres=1e-8, max_iters=200, use_smooth=True, 
                                            normalize=True, warm_start=False, post_scal=False, 
                                            scs_fallback=False, concurrent_update=False):
                    """Patched update_temporal that handles sparse.COO arrays properly."""
                    _original_csc_matrix = scipy.sparse.csc_matrix

                    class PatchedCSCMatrix(scipy.sparse.csc_matrix):
                        def __new__(cls, arg1, shape=None, dtype=None, copy=False):
                            if hasattr(arg1, 'todense'):
                                arg1 = arg1.todense()
                            return _original_csc_matrix(arg1, shape=shape, dtype=dtype, copy=copy)

                    scipy.sparse.csc_matrix = PatchedCSCMatrix
                    try:
                        result = _original_update_temporal(
                            A, C, b=b, f=f, Y=Y, YrA=YrA, noise_freq=noise_freq,
                            p=p, add_lag=add_lag, jac_thres=jac_thres, sparse_penal=sparse_penal,
                            bseg=bseg, med_wd=med_wd, zero_thres=zero_thres, max_iters=max_iters,
                            use_smooth=use_smooth, normalize=normalize, warm_start=warm_start,
                            post_scal=post_scal, scs_fallback=scs_fallback, 
                            concurrent_update=concurrent_update
                        )
                    finally:
                        scipy.sparse.csc_matrix = _original_csc_matrix
                    return result

                cnmf_module.update_temporal = patched_update_temporal

                # <<< NEW PATCH: scipy.linalg.solve_triangular NaN/Inf guard >>>
                # Prevents ValueError('array must not contain infs or NaNs') from
                # crashing Dask tasks and triggering infinite retry → OOM loops.
                from distributed import WorkerPlugin

                class SolveTriangularNaNGuard(WorkerPlugin):
                    """Patch scipy.linalg.solve_triangular on each Dask worker process
                    to sanitize NaN/Inf inputs instead of crashing."""
                    
                    def setup(self, worker):
                        import scipy.linalg
                        import numpy as np
                        
                        _original = scipy.linalg.solve_triangular
                        
                        def patched_solve_triangular(a, b, **kwargs):
                            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                            b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
                            return _original(a, b, **kwargs)
                        
                        scipy.linalg.solve_triangular = patched_solve_triangular

                logger.info("Applied minian compatibility patches (NetworkX, sparse arrays, dask.block, solve_triangular NaN guard)")

                # Now import minian modules (after patches are applied)
                from minian.cnmf import (
                    get_noise_fft,
                    unit_merge,
                    update_spatial,
                    update_temporal,
                    update_background,
                )
                from minian.initialization import (
                    initA,
                    initC,
                    ks_refine,
                    pnr_refine,
                    seeds_init,
                    seeds_merge,
                )
                from minian.motion_correction import apply_transform, estimate_motion
                from minian.preprocessing import denoise, remove_background
                from minian.utilities import (
                    TaskAnnotation,
                    get_optimal_chk,
                    load_videos,
                    save_minian,
                )
                from minian.visualization import write_video

                # ===== CONTAINER-AWARE MEMORY DETECTION =====
                def get_container_memory_limit():
                    """Get memory limit respecting container cgroups (v1 and v2)."""
                    try:
                        with open("/sys/fs/cgroup/memory.max") as f:
                            limit = f.read().strip()
                            if limit != "max":
                                return int(limit)
                    except (FileNotFoundError, PermissionError):
                        pass
                    try:
                        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                            limit = int(f.read().strip())
                            if limit < 9223372036854771712:
                                return limit
                    except (FileNotFoundError, PermissionError):
                        pass
                    return psutil.virtual_memory().total

                # ===== DASK CLUSTER CONFIGURATION =====
                memory_total = get_container_memory_limit()
                n_workers = int(
                    os.getenv(
                        "MINIAN_NWORKERS",
                        max(1, min(int(multiprocessing.cpu_count() * 0.4), 8)),
                    )
                )
                memory_per_worker = int(memory_total * 0.8 / n_workers)

                memory_limit_env = os.getenv("MINIAN_MEMORY_LIMIT")
                if memory_limit_env:
                    memory_limit = (
                        memory_limit_env
                        if any(c.isalpha() for c in memory_limit_env)
                        else f"{memory_limit_env}GB"
                    )
                else:
                    memory_limit = f"{memory_per_worker // (1024**3)}GB"

                # Set minian intermediate storage paths
                minian_data_path = str(output_dir / "minian_data")
                os.makedirs(minian_data_path, exist_ok=True)
                os.environ["MINIAN_INTERMEDIATE"] = minian_data_path

                # Helper function to clean intermediate files
                def clean_intermediate_files(directory_path, file_names):
                    """Remove intermediate zarr files to avoid conflicts."""
                    for fname in file_names:
                        path = os.path.join(directory_path, f"{fname}.zarr")
                        if os.path.exists(path):
                            shutil.rmtree(path)

                def sanitize_array(arr, name="array"):
                    """Replace NaN/Inf in an xarray DataArray with 0.
                    Works with both lazy (dask-backed) and in-memory arrays."""
                    logger.info(f"Sanitizing {name} (replacing NaN/Inf with 0)...")
                    if hasattr(arr.data, "dask"):
                        sanitized = xr.apply_ufunc(
                            lambda x: np.nan_to_num(
                                x, nan=0.0, posinf=0.0, neginf=0.0
                            ),
                            arr,
                            dask="parallelized",
                            output_dtypes=[arr.dtype],
                        )
                    else:
                        sanitized = xr.DataArray(
                            np.nan_to_num(
                                arr.values, nan=0.0, posinf=0.0, neginf=0.0
                            ),
                            coords=arr.coords,
                            dims=arr.dims,
                        )
                    return sanitized

                # Start Dask cluster
                logger.info(
                    f"Starting Minian processing with {n_workers} workers, "
                    f"{memory_limit} memory limit per worker..."
                )
                cluster = LocalCluster(
                    n_workers=n_workers,
                    memory_limit=memory_limit,
                    resources={"MEM": 1},
                    threads_per_worker=2,
                    dashboard_address=None,
                )
                annotation_plugin = TaskAnnotation()
                cluster.scheduler.add_plugin(annotation_plugin)
                client = Client(cluster)
                client.register_worker_plugin(SolveTriangularNaNGuard())

                try:
                    # ===== LOAD VIDEOS =====
                    logger.info("Loading videos...")
                    default_load_params = {
                        "pattern": r".*\.avi$",
                        "dtype": np.uint8,
                        "downsample": dict(frame=1, height=1, width=1),
                        "downsample_strategy": "subset",
                    }
                    param_load_videos = {
                        **default_load_params,
                        **params.get("param_load_videos", {}),
                    }
                    varr = load_videos(
                        str(pathlib.Path(avi_files[0]).parent), **param_load_videos
                    )
                    chk, _ = get_optimal_chk(varr, dtype=float)

                    # Save raw video to zarr
                    logger.info("Saving raw video to zarr...")
                    varr = save_minian(
                        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
                        minian_data_path,
                        overwrite=True,
                    )
                    logger.info(
                        f"Loaded video: {varr.sizes['frame']} frames, "
                        f"{varr.sizes['height']}x{varr.sizes['width']} pixels"
                    )

                    # ===== PREPROCESSING =====
                    logger.info("Preprocessing: glow removal...")
                    varr_min = varr.min("frame").compute()
                    varr_ref = varr - varr_min
                    varr_ref = varr_ref.clip(min=0)

                    logger.info("Preprocessing: denoising...")
                    param_denoise = params.get(
                        "param_denoise", {"method": "median", "ksize": 7}
                    )
                    varr_ref = denoise(varr_ref, **param_denoise)

                    logger.info("Preprocessing: background removal...")
                    param_background_removal = params.get(
                        "param_background_removal", {"method": "tophat", "wnd": 10}
                    )
                    varr_ref = remove_background(varr_ref, **param_background_removal)

                    # Keep it chunked
                    varr_ref = varr_ref.chunk({"frame": chk["frame"], "height": -1, "width": -1})

                    # ===== MOTION CORRECTION =====
                    logger.info("Estimating motion...")
                    param_estimate_motion = params.get(
                        "param_estimate_motion", {"dim": "frame"}
                    )
                    motion = estimate_motion(varr_ref, **param_estimate_motion)
                    motion = save_minian(
                        motion.rename("motion").chunk({"frame": chk["frame"]}),
                        minian_data_path,
                        overwrite=True,
                    )

                    logger.info("Applying motion correction...")
                    Y = apply_transform(varr_ref, motion, fill=0)

                    # <<< NEW: Sanitize after motion correction >>>
                    # apply_transform can introduce NaN at frame borders where the
                    # shift goes beyond the image boundary. The fill=0 param should
                    # handle this, but edge cases slip through on large datasets
                    # with float64 precision. This prevents downstream
                    # solve_triangular crashes in CNMF.
                    Y = sanitize_array(Y, "Y_post_motion_correction")

                    # Save two versions with different chunking
                    logger.info("Saving motion-corrected video (frame-chunked)...")
                    Y_fm_chk = save_minian(
                        Y.astype(np.float32).rename("Y_fm_chk"),
                        minian_data_path,
                        overwrite=True,
                    )

                    logger.info("Saving motion-corrected video (spatial-chunked)...")
                    Y_hw_chk = save_minian(
                        Y_fm_chk.rename("Y_hw_chk"),
                        minian_data_path,
                        overwrite=True,
                        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
                    )

                    # Save motion corrected video as mp4
                    logger.info("Writing motion corrected video...")
                    write_video(Y_fm_chk, "motion_corrected.mp4", str(output_dir))

                    # Create and save max projection
                    logger.info("Computing max projection...")
                    max_proj = Y_fm_chk.max("frame").compute()
                    max_proj = save_minian(max_proj.rename("max_proj"), minian_data_path, overwrite=True)

                    # ===== SEED INITIALIZATION =====
                    logger.info("Initializing seeds...")
                    param_seeds_init = params.get(
                        "param_seeds_init",
                        {
                            "wnd_size": 1000,
                            "method": "rolling",
                            "stp_size": 500,
                            "max_wnd": 15,
                            "diff_thres": 3,
                        },
                    )
                    seeds = seeds_init(Y_fm_chk, **param_seeds_init)
                    logger.info(f"Initial seeds: {len(seeds)}")

                    logger.info("Refining seeds with PNR...")
                    param_pnr_refine = params.get(
                        "param_pnr_refine", {"noise_freq": 0.06, "thres": 1}
                    )
                    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)
                    logger.info(f"Seeds after PNR refine: {seeds['mask_pnr'].sum()} / {len(seeds)}")

                    logger.info("Refining seeds with KS test...")
                    param_ks_refine = params.get("param_ks_refine", {"sig": 0.05})
                    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)
                    logger.info(f"Seeds after KS refine: {seeds['mask_ks'].sum()} / {len(seeds)}")

                    logger.info("Merging seeds...")
                    param_seeds_merge = params.get(
                        "param_seeds_merge",
                        {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06},
                    )
                    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
                    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)
                    n_seeds = seeds_final["mask_mrg"].sum()
                    logger.info(f"Seeds after merge: {n_seeds} / {len(seeds_final)}")

                    if n_seeds == 0:
                        raise ValueError(
                            "No seeds remaining after refinement. "
                            "Consider adjusting param_seeds_init or param_pnr_refine thresholds."
                        )

                    # ===== INITIALIZE A AND C =====
                    logger.info("Initializing spatial footprints (A)...")
                    param_initialize = params.get(
                        "param_initialize", {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
                    )
                    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
                    A_init = save_minian(A_init.rename("A_init"), minian_data_path, overwrite=True)
                    logger.info(f"A_init shape: {A_init.shape}")

                    logger.info("Initializing temporal traces (C)...")
                    C_init = initC(Y_fm_chk, A_init)
                    C_init = save_minian(
                        C_init.rename("C_init"),
                        minian_data_path,
                        overwrite=True,
                        chunks={"unit_id": 1, "frame": -1},
                    )
                    logger.info(f"C_init shape: {C_init.shape}")

                    # Initial unit merge
                    logger.info("Initial unit merge...")
                    param_init_merge = params.get("param_init_merge", {"thres_corr": 0.8})
                    A, C = unit_merge(A_init, C_init, **param_init_merge)
                    A = save_minian(A.rename("A"), minian_data_path, overwrite=True)
                    C = save_minian(C.rename("C"), minian_data_path, overwrite=True)
                    C_chk = save_minian(
                        C.rename("C_chk"),
                        minian_data_path,
                        overwrite=True,
                        chunks={"unit_id": -1, "frame": chk["frame"]},
                    )
                    logger.info(f"Units after initial merge: {A.sizes['unit_id']}")

                    # ===== INITIALIZE BACKGROUND =====
                    logger.info("Initializing background terms...")
                    b, f = update_background(Y_fm_chk, A, C_chk)
                    f = save_minian(f.rename("f"), minian_data_path, overwrite=True)
                    b = save_minian(b.rename("b"), minian_data_path, overwrite=True)
                    logger.info(f"Background initialized - b: {b.shape}, f: {f.shape}")

                    # ===== COMPUTE NOISE STATISTICS =====
                    logger.info("Computing noise statistics...")
                    param_get_noise = params.get("param_get_noise", {"noise_range": (0.06, 0.5)})
                    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
                    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), minian_data_path, overwrite=True)

                    # ========================================
                    # CNMF ITERATION 1
                    # ========================================

                    # ----- First Spatial Update -----
                    logger.info("CNMF Iteration 1: Spatial update...")
                    param_first_spatial = params.get(
                        "param_first_spatial",
                        {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)},
                    )
                    A_new, mask, norm_fac = update_spatial(
                        Y_hw_chk, A, C, sn_spatial, **param_first_spatial
                    )
                    C_new = save_minian(
                        (C.sel(unit_id=mask) * norm_fac).rename("C_new"),
                        minian_data_path,
                        overwrite=True,
                    )
                    C_chk_new = save_minian(
                        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
                        minian_data_path,
                        overwrite=True,
                    )
                    logger.info(f"Units after first spatial update: {A_new.sizes['unit_id']}")

                    # Update background after first spatial
                    logger.info("Updating background...")
                    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

                    # ----- First Temporal Update -----
                    logger.info("CNMF Iteration 1: Temporal update...")
                    param_first_temporal = params.get(
                        "param_first_temporal",
                        {
                            "noise_freq": 0.06,
                            "sparse_penal": 1,
                            "p": 1,
                            "add_lag": 20,
                            "jac_thres": 0.2,
                        },
                    )
                    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
                        A_new, C_new,
                        Y=Y_fm_chk,
                        b=b_new, f=f_new,
                        **param_first_temporal
                    )
                    logger.info(f"Units after first temporal update: {C_new.sizes['unit_id']}")
                    C_new = sanitize_array(C_new, "C_new_pre_merge")
                    S_new = sanitize_array(S_new, "S_new_pre_merge")
                    # ----- First Merge -----
                    logger.info("CNMF Iteration 1: Merging units...")
                    param_first_merge = params.get("param_first_merge", {"thres_corr": 0.8})

                    # <<< PATCHED: unit_merge coordinate fix
                    # update_temporal returns C_new/S_new already filtered by mask internally.
                    # Sync A with C_new's actual coordinates instead of using the boolean mask,
                    # which can cause alignment mismatches on large datasets.
                    A_filtered = A_new.sel(unit_id=C_new.coords["unit_id"].values)
                    logger.info(
                        f"Unit sync check — A: {A_filtered.sizes['unit_id']}, "
                        f"C: {C_new.sizes['unit_id']}, S: {S_new.sizes['unit_id']}"
                    )
                    A_mrg, C_mrg, [S_mrg] = unit_merge(
                        A_filtered, C_new, [S_new], **param_first_merge
                    )
                    # >>> END PATCH
                    logger.info(f"Units after first merge: {A_mrg.sizes['unit_id']}")
                    A_mrg = save_minian(A_mrg.rename("A_mrg"), minian_data_path, overwrite=True)
                    C_mrg = save_minian(C_mrg.rename("C_mrg"), minian_data_path, overwrite=True)
                    S_mrg = save_minian(S_mrg.rename("S_mrg"), minian_data_path, overwrite=True)
                    gc.collect()
                    logger.info("Saved merged results to zarr for iteration 2")

                    # ========================================
                    # CNMF ITERATION 2
                    # ========================================

                    # ----- Second Spatial Update -----
                    logger.info("CNMF Iteration 2: Spatial update...")
                    param_second_spatial = params.get(
                        "param_second_spatial",
                        {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)},
                    )
                    A_new2, mask2, norm_fac2 = update_spatial(
                        Y_hw_chk, A_mrg, C_mrg, sn_spatial, **param_second_spatial
                    )
                    C_new2 = C_mrg.sel(unit_id=mask2) * norm_fac2
                    logger.info(f"Units after second spatial update: {A_new2.sizes['unit_id']}")

                    # Update background after second spatial
                    logger.info("Updating background...")
                    b_new2, f_new2 = update_background(Y_fm_chk, A_new2, C_new2)

                    # ----- Second Temporal Update -----
                    logger.info("CNMF Iteration 2: Temporal update...")
                    param_second_temporal = params.get(
                        "param_second_temporal",
                        {
                            "noise_freq": 0.06,
                            "sparse_penal": 1,
                            "p": 1,
                            "add_lag": 20,
                            "jac_thres": 0.4,
                        },
                    )
                    C_final, S_final, b0_final, c0_final, g_final, mask_final = update_temporal(
                        A_new2, C_new2,
                        Y=Y_fm_chk,
                        b=b_new2, f=f_new2,
                        **param_second_temporal
                    )
                    C_final = sanitize_array(C_final, "C_final_post_temporal_2")
                    S_final = sanitize_array(S_final, "S_final_post_temporal_2")

                    # <<< PATCHED: unit_merge coordinate fix (same pattern as iteration 1)
                    # Sync A with C_final's actual coordinates
                    A_final = A_new2.sel(unit_id=C_final.coords["unit_id"].values)
                    # >>> END PATCH
                    logger.info(f"Final units: {A_final.sizes['unit_id']}")

                    # ===== SAVE FINAL RESULTS =====
                    logger.info("Saving final results to output directory...")
                    final_save_params = {"dpath": str(output_dir), "overwrite": True}

                    A_final = save_minian(A_final.rename("A"), **final_save_params)
                    C_final = save_minian(C_final.rename("C"), **final_save_params)
                    S_final = save_minian(S_final.rename("S"), **final_save_params)
                    b_final = save_minian(b_new2.rename("b"), **final_save_params)
                    f_final = save_minian(f_new2.rename("f"), **final_save_params)
                    motion_final = save_minian(motion.rename("motion"), **final_save_params)
                    max_proj_final = save_minian(max_proj.rename("max_proj"), **final_save_params)

                    logger.info(
                        f"Minian processing complete. {A_final.sizes['unit_id']} units detected."
                    )

                except Exception as e:
                    logger.error(f"Minian processing failed: {e}")
                    raise e
                finally:
                    client.close()
                    cluster.close()

                # Load results and prepare for insertion
                minian_loader = MinianLoader(minian_data_path)
                key["processing_time"] = minian_loader.creation_time

                # Get minian version if available
                try:
                    import minian
                    key["package_version"] = getattr(minian, "__version__", "")
                except (ImportError, AttributeError):
                    key["package_version"] = ""

                file_entries = [
                    {
                        **key,
                        "file_name": f.name,
                        "file": f.as_posix(),
                    }
                    for f in output_dir.rglob("*")
                    if f.is_file()
                ]

        else:
            raise ValueError(f"Unknown task mode: {task_mode}")
        return (file_entries, output_dir)

    def make_insert(self, key, file_entries, output_dir):
        # update processing_output_dir
        ProcessingTask.update1(
            {
                **key,
                "processing_output_dir": output_dir.relative_to(
                    get_processed_root_data_dir()
                ).as_posix(),
            }
        )
        self.insert1(dict(**key, processing_time=datetime.now(timezone.utc)))
        # for file in file_entries:
        #     self.File.insert1(file, ignore_extra_fields=True)


# Motion Correction --------------------------------------------------------------------


@schema
class MotionCorrection(dj.Imported):
    """Automated table performing motion correction analysis.

    Attributes:
        Processing (foreign key): Processing primary key.
        Channel.proj(motion_correct_channel='channel'): Channel used for motion correction.
    """

    definition = """
    -> Processing
    ---
    -> Channel.proj(motion_correct_channel='channel') # channel used for
                                                      # motion correction
    """

    class RigidMotionCorrection(dj.Part):
        """Details of rigid motion correction performed on the imaging data.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            outlier_frames (longblob): Mask with true for frames with outlier shifts
                (already corrected).
            y_shifts (longblob): y motion correction shifts (pixels).
            x_shifts (longblob): x motion correction shifts (pixels).
            z_shifts (longblob, optional): z motion correction shifts (z-drift, pixels).
            y_std (float): standard deviation of y shifts across all frames (pixels).
            x_std (float): standard deviation of x shifts across all frames (pixels).
            z_std (float, optional): standard deviation of z shifts across all frames
                (pixels).
        """

        definition = """# Details of rigid motion correction performed on the imaging data
        -> master
        ---
        outlier_frames=null : longblob  # mask with true for frames with outlier shifts (already corrected)
        y_shifts            : longblob  # (pixels) y motion correction shifts
        x_shifts            : longblob  # (pixels) x motion correction shifts
        z_shifts=null       : longblob  # (pixels) z motion correction shifts (z-drift)
        y_std               : float     # (pixels) standard deviation of y shifts across all frames
        x_std               : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null          : float     # (pixels) standard deviation of z shifts across all frames
        """

    class NonRigidMotionCorrection(dj.Part):
        """Piece-wise rigid motion correction - tile the FOV into multiple 3D
        blocks/patches.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            outlier_frames (longblob, null): Mask with true for frames with outlier
                shifts (already corrected).
            block_height (int): Block height in pixels.
            block_width (int): Block width in pixels.
            block_depth (int): Block depth in pixels.
            block_count_y (int): Number of blocks tiled in the y direction.
            block_count_x (int): Number of blocks tiled in the x direction.
            block_count_z (int): Number of blocks tiled in the z direction.
        """

        definition = """# Details of non-rigid motion correction performed on the imaging data
        -> master
        ---
        outlier_frames=null : longblob # mask with true for frames with outlier shifts (already corrected)
        block_height        : int      # (pixels)
        block_width         : int      # (pixels)
        block_depth         : int      # (pixels)
        block_count_y       : int      # number of blocks tiled in the y direction
        block_count_x       : int      # number of blocks tiled in the x direction
        block_count_z       : int      # number of blocks tiled in the z direction
        """

    class Block(dj.Part):
        """FOV-tiled blocks used for non-rigid motion correction.

        Attributes:
            NonRigidMotionCorrection (foreign key): Primary key from
                NonRigidMotionCorrection.
            block_id (int): Unique block ID.
            block_y (longblob): y_start and y_end in pixels for this block
            block_x (longblob): x_start and x_end in pixels for this block
            block_z (longblob): z_start and z_end in pixels for this block
            y_shifts (longblob): y motion correction shifts for every frame in pixels
            x_shifts (longblob): x motion correction shifts for every frame in pixels
            z_shift=null (longblob, optional): x motion correction shifts for every frame
                in pixels
            y_std (float): standard deviation of y shifts across all frames in pixels
            x_std (float): standard deviation of x shifts across all frames in pixels
            z_std=null (float, optional): standard deviation of z shifts across all frames
                in pixels
        """

        definition = """# FOV-tiled blocks used for non-rigid motion correction
        -> master.NonRigidMotionCorrection
        block_id        : int
        ---
        block_y         : longblob  # (y_start, y_end) in pixel of this block
        block_x         : longblob  # (x_start, x_end) in pixel of this block
        block_z         : longblob  # (z_start, z_end) in pixel of this block
        y_shifts        : longblob  # (pixels) y motion correction shifts for every frame
        x_shifts        : longblob  # (pixels) x motion correction shifts for every frame
        z_shifts=null   : longblob  # (pixels) z motion correction shifts for every frame
        y_std           : float     # (pixels) standard deviation of y shifts across all frames
        x_std           : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null      : float     # (pixels) standard deviation of z shifts across all frames
        """

    class Summary(dj.Part):
        """Summary images for each field and channel after corrections.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            ref_image (longblob): Image used as alignment template.
            average_image (longblob): Mean of registered frames.
            correlation_image (longblob, optional): Correlation map (computed during
                cell detection).
            max_proj_image (longblob, optional): Max of registered frames.
        """

        definition = """# Summary images for each field and channel after corrections
        -> master
        ---
        ref_image               : longblob  # image used as alignment template
        average_image           : longblob  # mean of registered frames
        correlation_image=null  : longblob  # correlation map (computed during cell detection)
        max_proj_image=null     : longblob  # max of registered frames
        """

    def make(self, key):
        """Populate tables with motion correction data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            self.insert1(
                {**key, "motion_correct_channel": caiman_dataset.alignment_channel}
            )

            # -- rigid motion correction --
            if caiman_dataset.is_pw_rigid:
                # -- non-rigid motion correction --
                (
                    nonrigid_correction,
                    nonrigid_blocks,
                ) = caiman_dataset.extract_pw_rigid_mc()
                nonrigid_correction.update(**key)
                self.NonRigidMotionCorrection.insert1(nonrigid_correction)
                self.Block.insert(
                    [{**block, **key} for block in nonrigid_blocks.values()]
                )
            else:
                # -- rigid motion correction --
                rigid_correction = caiman_dataset.extract_rigid_mc()
                rigid_correction.update(**key)
                self.RigidMotionCorrection.insert1(rigid_correction)

            # -- summary images --
            summary_images = {
                **key,
                "ref_image": caiman_dataset.ref_image.transpose(2, 0, 1),
                "average_image": caiman_dataset.mean_image.transpose(2, 0, 1),
                "correlation_image": caiman_dataset.correlation_map.transpose(2, 0, 1),
                "max_proj_image": caiman_dataset.max_proj_image.transpose(2, 0, 1),
            }
            self.Summary.insert1(summary_images)

        elif method == "minian":
            minian_dataset = loaded_result

            self.insert1(
                {**key, "motion_correct_channel": minian_dataset.alignment_channel}
            )

            # Minian uses rigid motion correction
            rigid_correction = minian_dataset.extract_rigid_mc()
            if rigid_correction is not None:
                rigid_correction.update(**key)
                self.RigidMotionCorrection.insert1(rigid_correction)

            # -- summary images --
            ref_image = minian_dataset.ref_image
            mean_image = minian_dataset.mean_image
            max_proj_image = minian_dataset.max_proj_image
            correlation_image = minian_dataset.correlation_map

            summary_images = {
                **key,
                "ref_image": (
                    ref_image if ref_image is not None else np.zeros((1, 1, 1))
                ),
                "average_image": (
                    mean_image if mean_image is not None else np.zeros((1, 1, 1))
                ),
                "correlation_image": correlation_image,
                "max_proj_image": max_proj_image,
            }
            self.Summary.insert1(summary_images)

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


# Segmentation -------------------------------------------------------------------------


@schema
class Segmentation(dj.Computed):
    """Automated table computes different mask segmentations.

    Attributes:
        Processing (foreign key): Processing primary key.
    """

    definition = """ # Different mask segmentations.
    -> Processing
    """

    class Mask(dj.Part):
        """Details of the masks identified from the Segmentation procedure.

        Attributes:
            Segmentation (foreign key): Primary key from Segmentation.
            mask (int): Unique mask ID.
            Channel.proj(segmentation_channel='channel') (foreign key): Channel
                used for segmentation.
            mask_npix (int): Number of pixels in ROIs.
            mask_center_x (int): Center x coordinate in pixel.
            mask_center_y (int): Center y coordinate in pixel.
            mask_center_z (int): Center z coordinate in pixel.
            mask_xpix (longblob): X coordinates in pixels.
            mask_ypix (longblob): Y coordinates in pixels.
            mask_zpix (longblob): Z coordinates in pixels.
            mask_weights (longblob): Weights of the mask at the indices above.
        """

        definition = """ # A mask produced by segmentation.
        -> master
        mask               : smallint
        ---
        -> Channel.proj(segmentation_channel='channel')  # channel used for segmentation
        mask_npix          : int       # number of pixels in ROIs
        mask_center_x      : int       # center x coordinate in pixel
        mask_center_y      : int       # center y coordinate in pixel
        mask_center_z=null : int       # center z coordinate in pixel
        mask_xpix          : longblob  # x coordinates in pixels
        mask_ypix          : longblob  # y coordinates in pixels
        mask_zpix=null     : longblob  # z coordinates in pixels
        mask_weights       : longblob  # weights of the mask at the indices above
        """

    def make(self, key):
        """Populates table with segmentation data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", caiman_dataset.segmentation_channel
            )

            masks, cells = [], []
            for mask in caiman_dataset.masks:
                masks.append(
                    {
                        **key,
                        "segmentation_channel": segmentation_channel,
                        "mask": mask["mask_id"],
                        "mask_npix": mask["mask_npix"],
                        "mask_center_x": mask["mask_center_x"],
                        "mask_center_y": mask["mask_center_y"],
                        "mask_center_z": mask["mask_center_z"],
                        "mask_xpix": mask["mask_xpix"],
                        "mask_ypix": mask["mask_ypix"],
                        "mask_zpix": mask["mask_zpix"],
                        "mask_weights": mask["mask_weights"],
                    }
                )
                if mask["accepted"]:
                    cells.append(
                        {
                            **key,
                            "mask_classification_method": "caiman_default_classifier",
                            "mask": mask["mask_id"],
                            "mask_type": "soma",
                        }
                    )

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1(
                    {
                        **key,
                        "mask_classification_method": "caiman_default_classifier",
                    },
                    allow_direct_insert=True,
                )
                MaskClassification.MaskType.insert(
                    cells, ignore_extra_fields=True, allow_direct_insert=True
                )

        elif method == "minian":
            minian_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from minian loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", minian_dataset.segmentation_channel
            )

            masks, cells = [], []
            for mask in minian_dataset.masks:
                masks.append(
                    {
                        **key,
                        "segmentation_channel": segmentation_channel,
                        "mask": mask["mask_id"],
                        "mask_npix": mask["mask_npix"],
                        "mask_center_x": mask["mask_center_x"],
                        "mask_center_y": mask["mask_center_y"],
                        "mask_center_z": mask["mask_center_z"],
                        "mask_xpix": mask["mask_xpix"],
                        "mask_ypix": mask["mask_ypix"],
                        "mask_zpix": mask["mask_zpix"],
                        "mask_weights": mask["mask_weights"],
                    }
                )
                if mask["accepted"]:
                    cells.append(
                        {
                            **key,
                            "mask_classification_method": "minian_default_classifier",
                            "mask": mask["mask_id"],
                            "mask_type": "soma",
                        }
                    )

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1(
                    {
                        **key,
                        "mask_classification_method": "minian_default_classifier",
                    },
                    allow_direct_insert=True,
                )
                MaskClassification.MaskType.insert(
                    cells, ignore_extra_fields=True, allow_direct_insert=True
                )

        else:
            raise NotImplementedError(f"Unknown/unimplemented method: {method}")


@schema
class MaskClassificationMethod(dj.Lookup):
    """Method to classify segmented masks.

    Attributes:
        mask_classification_method (foreign key, varchar(48) ): Method by which masks
            are classified into mask types.
    """

    definition = """
    mask_classification_method: varchar(48)
    """

    contents = zip(["caiman_default_classifier", "minian_default_classifier"])


@schema
class MaskClassification(dj.Computed):
    """Automated table with mask classification data.

    Attributes:
        Segmentation (foreign key): Segmentation primary key.
        MaskClassificationMethod (foreign key): MaskClassificationMethod primary key.
    """

    definition = """
    -> Segmentation
    -> MaskClassificationMethod
    """

    class MaskType(dj.Part):
        """Automated table storing mask type data.

        Attributes:
            MaskClassification (foreign key): MaskClassification primary key.
            Segmentation.Mask (foreign key): Segmentation.Mask primary key.
            MaskType (dict): Select mask type from entries within `MaskType` look up table.
            confidence (float): Statistical confidence of mask classification.
        """

        definition = """
        -> master
        -> Segmentation.Mask
        ---
        -> MaskType
        confidence=null: float
        """

    def make(self, key):
        raise NotImplementedError(
            "To add to this table, use `insert` with allow_direct_insert=True"
        )


# Fluorescence & Activity Traces -------------------------------------------------------


@schema
class Fluorescence(dj.Computed):
    """Extracts fluorescence trace information.

    Attributes:
        Segmentation (foreign key): Segmentation primary key.
    """

    definition = """  # fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
        """Automated table with Fluorescence traces

        Attributes:
            Fluorescence (foreign key): Fluorescence primary key.
            Segmentation.Mask (foreign key): Segmentation.Mask primary key.
            Channel.proj(fluorescence_channel='channel') (foreign key, query): Channel
                used for this trace.
            fluorescence (longblob): A fluorescence trace associated with a given mask.
            neuropil_fluorescence (longblob): A neuropil fluorescence trace.
        """

        definition = """
        -> master
        -> Segmentation.Mask
        -> Channel.proj(fluorescence_channel='channel')  # channel used for this trace
        ---
        fluorescence                : longblob  # fluorescence trace associated
                                                # with this mask
        neuropil_fluorescence=null  : longblob  # Neuropil fluorescence trace
        """

    def make(self, key):
        """Populates table with fluorescence trace data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", caiman_dataset.segmentation_channel
            )

            fluo_traces = []
            for mask in caiman_dataset.masks:
                fluo_traces.append(
                    {
                        **key,
                        "mask": mask["mask_id"],
                        "fluorescence_channel": segmentation_channel,
                        "fluorescence": mask["inferred_trace"],
                    }
                )

            self.insert1(key)
            self.Trace.insert(fluo_traces)

        elif method == "minian":
            minian_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from minian loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", minian_dataset.segmentation_channel
            )

            fluo_traces = []
            for mask in minian_dataset.masks:
                fluo_traces.append(
                    {
                        **key,
                        "mask": mask["mask_id"],
                        "fluorescence_channel": segmentation_channel,
                        "fluorescence": mask["inferred_trace"],
                    }
                )

            self.insert1(key)
            self.Trace.insert(fluo_traces)

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    """Lookup table for activity extraction methods.

    Attributes:
        extraction_method (foreign key, varchar(32) ): Extraction method from CaImAn or Minian.
    """

    definition = """
    extraction_method: varchar(32)
    """

    contents = zip(["caiman_deconvolution", "caiman_dff", "minian_deconvolution"])


@schema
class Activity(dj.Computed):
    """Inferred neural activity from the fluorescence trace.

    Attributes:
        Fluorescence (foreign key): Fluorescence primary key.
        ActivityExtractionMethod (foreign key): ActivityExtractionMethod primary key.
    """

    definition = """
    # inferred neural activity from fluorescence trace - e.g. dff, spikes
    -> Fluorescence
    -> ActivityExtractionMethod
    """

    class Trace(dj.Part):
        """Automated table with activity traces.

        Attributes:
            Activity (foreign key): Activity primary key.
            Fluorescence.Trace (foreign key): fluorescence.Trace primary key.
            activity_trace (longblob): Inferred activity trace.
        """

        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        activity_trace: longblob
        """

    @property
    def key_source(self):
        """Defines the order of keys when the `make` function is called."""
        caiman_key_source = (
            Fluorescence
            * ActivityExtractionMethod
            * ProcessingParamSet.proj("processing_method")
            & 'processing_method = "caiman"'
            & 'extraction_method LIKE "caiman%"'
        )

        minian_key_source = (
            Fluorescence
            * ActivityExtractionMethod
            * ProcessingParamSet.proj("processing_method")
            & 'processing_method = "minian"'
            & 'extraction_method LIKE "minian%"'
        )

        return caiman_key_source.proj() + minian_key_source.proj()

    def make(self, key):
        """Populates table with activity trace data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            if key["extraction_method"] in (
                "caiman_deconvolution",
                "caiman_dff",
            ):
                attr_mapper = {
                    "caiman_deconvolution": "spikes",
                    "caiman_dff": "dff",
                }

                # infer "segmentation_channel" - from params if available, else from caiman loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
                segmentation_channel = params.get(
                    "segmentation_channel", caiman_dataset.segmentation_channel
                )

                self.insert1(key)
                self.Trace.insert(
                    dict(
                        key,
                        mask=mask["mask_id"],
                        fluorescence_channel=segmentation_channel,
                        activity_trace=mask[attr_mapper[key["extraction_method"]]],
                    )
                    for mask in caiman_dataset.masks
                )

        elif method == "minian":
            minian_dataset = loaded_result

            if key["extraction_method"] == "minian_deconvolution":
                # infer "segmentation_channel" - from params if available, else from minian loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
                segmentation_channel = params.get(
                    "segmentation_channel", minian_dataset.segmentation_channel
                )

                self.insert1(key)
                self.Trace.insert(
                    dict(
                        key,
                        mask=mask["mask_id"],
                        fluorescence_channel=segmentation_channel,
                        activity_trace=mask["spikes"],
                    )
                    for mask in minian_dataset.masks
                    if "spikes" in mask
                )

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


@schema
class ProcessingQualityMetrics(dj.Computed):
    """Quality metrics used to evaluate the results of the calcium imaging analysis pipeline.

    Attributes:
        Fluorescence (foreign key): Primary key from Fluorescence.
    """

    definition = """
    -> Fluorescence
    """

    class Trace(dj.Part):
        """Quality metrics used to evaluate the fluorescence traces.

        Attributes:
            Fluorescence (foreign key): Primary key from Fluorescence.
            Fluorescence.Trace (foreign key): Primary key from Fluorescence.Trace.
            skewness (float): Skewness of the fluorescence trace.
            variance (float): Variance of the fluorescence trace.
        """

        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        skewness: float   # Skewness of the fluorescence trace.
        variance: float   # Variance of the fluorescence trace.
        """

    def make(self, key):
        """Populate the ProcessingQualityMetrics table and its part tables."""
        from scipy.stats import skew

        (
            fluorescence,
            fluorescence_channels,
            mask_ids,
        ) = (
            Segmentation.Mask * RecordingInfo * Fluorescence.Trace & key
        ).fetch("fluorescence", "fluorescence_channel", "mask")

        fluorescence = np.stack(fluorescence)

        self.insert1(key)

        self.Trace.insert(
            dict(
                key,
                fluorescence_channel=fluorescence_channel,
                mask=mask_id,
                skewness=skewness,
                variance=variance,
            )
            for fluorescence_channel, mask_id, skewness, variance in zip(
                fluorescence_channels,
                mask_ids,
                skew(fluorescence, axis=1),
                fluorescence.std(axis=1),
            )
        )


# Helper Functions ---------------------------------------------------------------------


class MinianLoader:
    """Loader class for Minian analysis results.

    Provides a consistent interface for accessing Minian outputs similar to CaImAn loader.
    """

    def __init__(self, output_dir):
        """Initialize the MinianLoader.

        Args:
            output_dir: Path to the directory containing Minian zarr outputs.
        """
        from minian.utilities import open_minian

        self.output_dir = pathlib.Path(output_dir)
        self._minian_ds = open_minian(str(self.output_dir))

        # Load core arrays
        self._A = self._minian_ds.get(
            "A"
        )  # Spatial footprints (unit_id, height, width)
        self._C = self._minian_ds.get("C")  # Temporal traces (unit_id, frame)
        self._S = self._minian_ds.get(
            "S"
        )  # Deconvolved activity/spikes (unit_id, frame)
        self._b = self._minian_ds.get("b")  # Background spatial (height, width)
        self._f = self._minian_ds.get("f")  # Background temporal (frame)
        self._b0 = self._minian_ds.get("b0")  # Baseline (unit_id, frame)
        self._c0 = self._minian_ds.get("c0")  # Initial calcium (unit_id, frame)

        # Try to load motion correction data
        self._motion = self._minian_ds.get("motion")

        # Try to load reference/max projection images
        self._max_proj = self._minian_ds.get("max_proj")
        self._varr_ref = self._minian_ds.get("varr_ref")

    @property
    def minian_dataset(self):
        """Return the raw Minian xarray Dataset."""
        return self._minian_ds

    @property
    def creation_time(self):
        """Get the creation time of the Minian output."""
        # Use the modification time of the output directory
        return datetime.fromtimestamp(self.output_dir.stat().st_mtime, tz=timezone.utc)

    @property
    def alignment_channel(self):
        """Channel used for motion correction (default 0 for miniscope)."""
        return 0

    @property
    def segmentation_channel(self):
        """Channel used for segmentation (default 0 for miniscope)."""
        return 0

    @property
    def is_pw_rigid(self):
        """Minian uses rigid motion correction by default."""
        return False

    @property
    def motion_shifts(self):
        """Return motion correction shifts as dict with 'x' and 'y' keys."""
        if self._motion is not None:
            motion_data = self._motion.compute()
            return {
                "x": motion_data.sel(shift_dim="width").values,
                "y": motion_data.sel(shift_dim="height").values,
            }
        return None

    def extract_rigid_mc(self):
        """Extract rigid motion correction data in format compatible with MotionCorrection table."""
        shifts = self.motion_shifts
        if shifts is None:
            return None

        return {
            "x_shifts": shifts["x"],
            "y_shifts": shifts["y"],
            "x_std": np.std(shifts["x"]),
            "y_std": np.std(shifts["y"]),
        }

    @property
    def ref_image(self):
        """Return reference image used for motion correction."""
        if self._varr_ref is not None:
            # Take mean across frames for reference
            return self._varr_ref.mean(dim="frame").compute().values[np.newaxis, :, :]
        return None

    @property
    def mean_image(self):
        """Return mean image (average across frames)."""
        if self._varr_ref is not None:
            return self._varr_ref.mean(dim="frame").compute().values[np.newaxis, :, :]
        return None

    @property
    def max_proj_image(self):
        """Return maximum projection image."""
        if self._max_proj is not None:
            return self._max_proj.compute().values[np.newaxis, :, :]
        elif self._varr_ref is not None:
            return self._varr_ref.max(dim="frame").compute().values[np.newaxis, :, :]
        return None

    @property
    def correlation_map(self):
        """Return correlation image (computed during initialization)."""
        # Minian doesn't store correlation image by default
        # Return None or compute if needed
        return None

    @property
    def masks(self):
        """Extract mask information in format compatible with Segmentation table.

        Yields dict for each unit with mask properties.
        """
        if self._A is None:
            return []

        A_data = self._A.compute()
        C_data = self._C.compute() if self._C is not None else None
        S_data = self._S.compute() if self._S is not None else None

        masks = []
        for unit_idx, unit_id in enumerate(A_data.coords["unit_id"].values):
            footprint = A_data.sel(unit_id=unit_id).values

            # Find non-zero pixels
            mask_indices = np.where(footprint > 0)
            if len(mask_indices[0]) == 0:
                continue

            y_pix = mask_indices[0]
            x_pix = mask_indices[1]
            weights = footprint[y_pix, x_pix]

            mask_dict = {
                "mask_id": int(unit_id),
                "mask_npix": len(x_pix),
                "mask_center_x": int(np.mean(x_pix)),
                "mask_center_y": int(np.mean(y_pix)),
                "mask_center_z": None,
                "mask_xpix": x_pix,
                "mask_ypix": y_pix,
                "mask_zpix": None,
                "mask_weights": weights,
                "accepted": True,  # All units accepted (no manual curation)
            }

            # Add trace data if available
            if C_data is not None:
                mask_dict["inferred_trace"] = C_data.sel(unit_id=unit_id).values
            if S_data is not None:
                mask_dict["spikes"] = S_data.sel(unit_id=unit_id).values

            masks.append(mask_dict)

        return masks

    @property
    def num_units(self):
        """Return number of detected units."""
        if self._A is not None:
            return len(self._A.coords["unit_id"])
        return 0


def get_loader_result(key, table, full_output_dir=None) -> tuple:
    """Retrieve the loaded processed imaging results from the loader (e.g. caiman, etc.)

    Args:
        key (dict): the `key` to one entry of ProcessingTask.
        table (str): the class defining the table to retrieve
            the loaded results from (e.g. ProcessingTask).

    Returns:
        method, loaded_output (tuple): method string and loader object with results (e.g. caiman.CaImAn, etc.)
    """

    if full_output_dir is None:
        output_dir = (ProcessingParamSet * table & key).fetch1("processing_output_dir")
        output_dir = find_full_path(get_processed_root_data_dir(), output_dir)
    else:
        output_dir = full_output_dir
    method = (ProcessingParamSet * table & key).fetch1("processing_method")
    if method == "caiman":
        from element_interface import caiman_loader

        loaded_output = caiman_loader.CaImAn(output_dir)
    elif method == "minian":
        loaded_output = MinianLoader(output_dir)
    else:
        raise NotImplementedError("Unknown/unimplemented method: {}".format(method))

    return method, loaded_output
