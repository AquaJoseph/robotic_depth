"""Build marker_frames.tfrecord from local session folders without interaction.
Run beside dataset/: python marker_dataset.py
Dependencies: numpy, pandas, opencv-python, tensorflow.
"""
# %% Settings and frame indexing
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_ROOT = Path("dataset")
OUTPUT = Path("marker_frames.tfrecord")
RIM_Y = 250
MAX_DEPTH_RAW = 5000.0
# Set to "absolute" for raw sensor depth instead of signed reference-relative depth.
DEPTH_MODE = "displacement"

# Approximate reference-frame pixels from frame 000002. One row per channel.
# Convention: four inner markers, then eight outer markers, clockwise in each ring.
# Assumes a comparable camera pose between sessions. The centre disc is excluded.
TEMPLATE = np.array([
    [261, 96], [287, 151], [226, 179], [192, 125],
    [217, 41], [279, 47], [332, 86], [322, 174],
    [278, 203], [222, 218], [171, 176], [147, 97],
], dtype=np.float32)

def index_frames(root):
    records = []
    for manifest in sorted(root.glob("session_*/frames.csv")):
        folder = manifest.parent
        table = pd.read_csv(manifest).sort_values("frame_id")
        if table["frame_id"].duplicated().any():
            raise ValueError(f"Duplicate frame ID in {manifest}")
        for row in table.itertuples(index=False):
            records.append({
                "session": folder.name,
                "frame_id": int(row.frame_id),
                "rgb_path": folder / row.rgb,
                "depth_path": folder / row.depth,
            })
    if not records:
        raise FileNotFoundError(f"No session frames found under {root.resolve()}")
    return records


# %% Detect disc centres in original image coordinates
def detect_markers(gray, threshold=210, min_area=20, max_area=180):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or 4 * np.pi * area / perimeter**2 < 0.60:
            continue
        moments = cv2.moments(contour)
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
        if y < RIM_Y:
            centres.append((x, y))
    return np.asarray(centres, dtype=np.float32).reshape(-1, 2)



# %% Match candidates and read the sensor data
def match(prior, candidates, gate):
    """Unique matches only. A missing/ambiguous match gets NaN coordinates."""
    distances = np.linalg.norm(prior[:, None] - candidates[None, :], axis=2)
    near = distances <= gate
    unique = near & (near.sum(axis=0) == 1) & (near.sum(axis=1)[:, None] == 1)
    marker, detected = np.nonzero(unique)
    result = np.full((12, 2), np.nan, dtype=np.float32)
    result[marker] = candidates[detected]
    return result


def read_gray(record):
    image = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record["rgb_path"])
    if image.shape[:2] != (270, 480):
        raise ValueError(f"Unexpected image dimensions: {record['rgb_path']}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sample_depth(record, xy):
    depth = np.squeeze(np.load(record["depth_path"], allow_pickle=False))
    if depth.shape != (270, 480):
        raise ValueError(f"Unexpected depth dimensions: {record['depth_path']}")
    seen = np.isfinite(xy).all(axis=1)
    x, y = np.rint(xy[seen]).astype(int).T
    z = np.full(12, np.nan, dtype=np.float32)
    z[seen] = depth[y, x]
    z[(z <= 0) | (z > MAX_DEPTH_RAW)] = np.nan
    return z


def track_session(records):
    # Anchor both time directions to the earliest frame matching the reference layout.
    detections = [detect_markers(read_gray(r)) for r in records]
    tracks = np.full((len(records), 12, 2), np.nan, dtype=np.float32)
    for ref, candidates in enumerate(detections):
        seed = match(TEMPLATE, candidates, gate=18)
        if np.isfinite(seed).all():
            break
    else:
        raise ValueError(f"No frame in {records[0]['session']} matches the 12-marker template. "
                         "Check the camera pose and TEMPLATE coordinates.")

    tracks[ref] = seed
    for indices in (range(ref + 1, len(records)), range(ref - 1, -1, -1)):
        prior = seed.copy()
        for i in indices:
            tracks[i] = match(prior, detections[i], gate=12)
            seen = np.isfinite(tracks[i]).all(axis=1)
            # After a miss, allow recovery near the known reference location.
            prior = np.where(seen[:, None], tracks[i], seed)
    return ref, tracks


# %% Convert positions into dense model targets
def make_targets(xy, z, baseline, mode):
    seen = np.isfinite(xy).all(axis=1)
    # At output resolution: 480x270 -> 128x72, plus 28 rows on either side.
    uv = (np.where(seen[:, None], xy, 0) + 0.5) * (128 / 480) - 0.5
    uv[:, 1] += 28
    yy, xx = np.ogrid[:128, :128]
    dist2 = (xx[..., None] - uv[:, 0])**2 + (yy[..., None] - uv[:, 1])**2
    heatmaps = (np.exp(-dist2 / 4.5) * seen).astype(np.float32)
    if mode == "displacement":
        value = z - baseline
    elif mode == "absolute":
        value = z
    else:
        raise ValueError(f"Unknown depth mode: {mode}")
    mask = (dist2 <= 4) & seen & np.isfinite(value)
    depths = np.where(mask, value, 0).astype(np.float32)
    return heatmaps, depths, seen, mask


# %% Export and read the compressed TFRecord
def bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def export(root=DATA_ROOT, output=OUTPUT, mode=DEPTH_MODE):
    records = index_frames(root)
    output = Path(output)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".partial", delete=False) as f:
        temp = Path(f.name)
    try:
        with tf.io.TFRecordWriter(str(temp), options="GZIP") as writer:
            for session in sorted({r["session"] for r in records}):
                rows = [r for r in records if r["session"] == session]
                ref, tracks = track_session(rows)
                baseline = sample_depth(rows[ref], tracks[ref])
                supervised_z = 0
                for r, xy in zip(rows, tracks):
                    gray = tf.image.convert_image_dtype(read_gray(r)[..., None], tf.float32)
                    image = tf.image.resize_with_pad(gray, 256, 256)
                    heat, depths, seen, mask = make_targets(xy, sample_depth(r, xy), baseline, mode)
                    supervised_z += int(mask.any(axis=(0, 1)).sum())
                    tensors = dict(image=image, heatmaps=heat, depths=depths,
                                   visible=seen, depth_mask=mask)
                    features = {k: bytes_feature(tf.io.serialize_tensor(v).numpy())
                                for k, v in tensors.items()}
                    features.update(session=bytes_feature(session.encode()),
                                    frame_id=int_feature(r["frame_id"]),
                                    reference_frame=int_feature(rows[ref]["frame_id"]),
                                    depth_mode=bytes_feature(mode.encode()))
                    writer.write(tf.train.Example(features=tf.train.Features(
                        feature=features)).SerializeToString())
                print(f"{session}: {len(rows)} frames, reference {rows[ref]['frame_id']}, "
                      f"XY coverage {np.isfinite(tracks[..., 0]).mean():.1%}, "
                      f"Z coverage {supervised_z / tracks.shape[0] / 12:.1%}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Wrote {len(records)} frames to {output.resolve()}")


TENSOR_FIELDS = {
    "image": (tf.float32, (256, 256, 1)),
    "heatmaps": (tf.float32, (128, 128, 12)),
    "depths": (tf.float32, (128, 128, 12)),
    "visible": (tf.bool, (12,)),
    "depth_mask": (tf.bool, (128, 128, 12)),
}


def load_dataset(path=OUTPUT):
    schema = {k: tf.io.FixedLenFeature([], tf.string) for k in TENSOR_FIELDS}
    for k in ("session", "depth_mode"):
        schema[k] = tf.io.FixedLenFeature([], tf.string)
    for k in ("frame_id", "reference_frame"):
        schema[k] = tf.io.FixedLenFeature([], tf.int64)

    def parse(serialized):
        item = tf.io.parse_single_example(serialized, schema)
        for key, (dtype, shape) in TENSOR_FIELDS.items():
            item[key] = tf.ensure_shape(tf.io.parse_tensor(item[key], dtype), shape)
        return item

    return tf.data.TFRecordDataset(str(path), compression_type="GZIP").map(parse)


def for_model(item):
    targets = {key: item[key] for key in ("heatmaps", "depths")}
    weights = {"heatmaps": tf.cast(item["visible"][None, None, :], tf.float32),
               "depths": tf.cast(item["depth_mask"], tf.float32)}
    return item["image"], targets, weights


def pixel_squared_error(y_true, y_pred):
    # For model.compile(loss=...) so weights apply before channel reduction.
    return tf.math.squared_difference(y_true, y_pred)


# %% Build the output file; imports only expose the functions above
if __name__ == "__main__":
    export()
    dataset = load_dataset()
    first = next(iter(dataset))
    print({k: first[k].shape for k in ("image", "heatmaps", "depths")})
