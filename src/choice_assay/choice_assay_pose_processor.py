from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from expidite_rpi.core import api, file_naming
from expidite_rpi.core import configuration as root_cfg
from expidite_rpi.core.dp import DataProcessor
from expidite_rpi.core.dp_config_objects import DataProcessorCfg, Stream

logger = root_cfg.setup_logger("choice_assay")

CA_POSE_DATA_TYPE_ID = "CAPOSE"
CA_POSE_STREAM_INDEX: int = 0


def _pose_field_names(keypoint_count: int) -> list[str]:
    fields: list[str] = [
        "source_filename",
        "source_data_type_id",
        "source_stream_index",
        "frame_index",
    ]
    for idx in range(keypoint_count):
        fields.extend([f"kpt{idx}_x", f"kpt{idx}_y", f"kpt{idx}_conf"])
    return fields


@dataclass
class ChoiceAssayPoseProcessorCfg(DataProcessorCfg):
    model_path: str | Path
    keypoint_count: int = 7


DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG = ChoiceAssayPoseProcessorCfg(
    description="YOLO pose processor for trapcam sub-videos",
    outputs=[
        Stream(
            description="Pose keypoints per frame for trapcam clips",
            type_id=CA_POSE_DATA_TYPE_ID,
            index=CA_POSE_STREAM_INDEX,
            format=api.FORMAT.DF,
            fields=_pose_field_names(7),
        ),
    ],
    model_path=str(Path(__file__).resolve().parent.parent / "resources" / "yolo_pose.ncnn"),
    keypoint_count=7,
)


class ChoiceAssayPoseProcessor(DataProcessor):
    def __init__(self, config: ChoiceAssayPoseProcessorCfg, sensor_index: int) -> None:
        super().__init__(config, sensor_index)
        self.config = config
        self.model = self._load_model()

    def _load_model(self) -> object:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment-specific
            msg = "Ultralytics is required for pose inference"
            raise ImportError(msg) from exc

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            msg = f"Pose model not found at {model_path}"
            raise FileNotFoundError(msg)

        return YOLO(str(model_path))

    def _select_keypoints(self, result: object, keypoint_count: int) -> np.ndarray | None:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or len(keypoints) == 0:
            return None

        kpt_data = keypoints.data
        if hasattr(kpt_data, "cpu"):
            kpt_data = kpt_data.cpu().numpy()

        if kpt_data.ndim != 3:
            return None

        if getattr(result, "boxes", None) is not None and result.boxes is not None:
            conf = getattr(result.boxes, "conf", None)
            if conf is not None and len(conf) > 0:
                best_idx = int(conf.argmax().item())
            else:
                best_idx = 0
        else:
            best_idx = 0

        selected = kpt_data[best_idx]
        if selected.shape[0] < keypoint_count:
            pad = np.full((keypoint_count - selected.shape[0], 3), np.nan, dtype=float)
            selected = np.vstack([selected, pad])
        elif selected.shape[0] > keypoint_count:
            selected = selected[:keypoint_count]
        return selected

    def _frame_to_row(
        self,
        frame_index: int,
        keypoints: np.ndarray | None,
        source_filename: str,
        source_data_type_id: str,
        source_stream_index: int,
    ) -> dict:
        row = {
            "source_filename": source_filename,
            "source_data_type_id": source_data_type_id,
            "source_stream_index": source_stream_index,
            "frame_index": frame_index,
        }

        if keypoints is None:
            for idx in range(self.config.keypoint_count):
                row[f"kpt{idx}_x"] = np.nan
                row[f"kpt{idx}_y"] = np.nan
                row[f"kpt{idx}_conf"] = np.nan
            return row

        for idx in range(self.config.keypoint_count):
            row[f"kpt{idx}_x"] = float(keypoints[idx, 0])
            row[f"kpt{idx}_y"] = float(keypoints[idx, 1])
            row[f"kpt{idx}_conf"] = float(keypoints[idx, 2])
        return row

    def _process_video_file(self, video_path: Path) -> pd.DataFrame:
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            msg = f"Unable to open video: {video_path}"
            raise ValueError(msg)

        try:
            parts = file_naming.parse_record_filename(video_path)
            start_time = parts.get(api.RECORD_ID.TIMESTAMP.value)
            source_data_type_id = parts.get(api.RECORD_ID.DATA_TYPE_ID.value, "")
            source_stream_index = int(parts.get(api.RECORD_ID.STREAM_INDEX.value, -1))
            fps = float(video.get(cv2.CAP_PROP_FPS) or 0)

            rows: list[dict] = []
            frame_index = 0

            while True:
                ok, frame = video.read()
                if not ok:
                    break

                results = self.model(frame, verbose=False)
                result = results[0] if results else None
                keypoints = self._select_keypoints(result, self.config.keypoint_count) if result else None
                row = self._frame_to_row(
                    frame_index,
                    keypoints,
                    video_path.name,
                    source_data_type_id,
                    source_stream_index,
                )

                if start_time is not None and fps > 0:
                    row[api.RECORD_ID.TIMESTAMP.value] = start_time + timedelta(seconds=frame_index / fps)

                rows.append(row)
                frame_index += 1
        finally:
            video.release()

        return pd.DataFrame(rows)

    def process_data(self, input_data: pd.DataFrame | list[Path]) -> None:
        assert isinstance(input_data, list), f"Expected list of files, got {type(input_data)}"
        files: list[Path] = input_data  # type: ignore[invalid-assignment]
        results: list[pd.DataFrame] = []

        for f in files:
            try:
                result = self._process_video_file(f)
                results.append(result)
            except Exception:
                logger.exception(f"{root_cfg.RAISE_WARN()}Exception occurred processing video {f!s}")

        output_df = pd.concat(results) if results else pd.DataFrame()
        self.save_data(stream_index=CA_POSE_STREAM_INDEX, sensor_data=output_df)
