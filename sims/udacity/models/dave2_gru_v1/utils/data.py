import os
import tensorflow as tf
import numpy as np
import pandas as pd

class DrivingSequenceDataset:
    """
    Builds a tf.data.Dataset of (T,H,W,C) sequences for CNN+GRU training.

    CSV columns required:
      - 'image_filename'
      - 'predicted_steering_angle'
      - 'predicted_throttle'
    Optional columns (if present, sequences won't cross boundaries):
      - 'episode' or 'run_id'
      - 'frame_index' (used for stable sorting within an episode)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        seq_len: int = 5,
        seq_stride: int = 1,
        label_from: str = "last",
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        assert label_from in ("last", "center")
        self.img_dir    = img_dir
        self.seq_len    = seq_len
        self.seq_stride = seq_stride
        self.label_from = label_from
        self.batch_size = batch_size
        self.shuffle    = shuffle

        self.seq_paths, self.seq_labels = self._build_sequences(df)

    def _build_sequences(self, df: pd.DataFrame):
        df = df.copy()

        # Choose grouping for episode/run boundaries
        group_col = None
        for cand in ("episode", "run_id"):
            if cand in df.columns:
                group_col = cand
                break

        if group_col:
            groups = [g for _, g in df.groupby(group_col, sort=False)]
        else:
            groups = [df]

        seq_paths, seq_labels = [], []

        for g in groups:
            # Stable order within a group
            if "frame_index" in g.columns:
                g = g.sort_values("frame_index")
            else:
                g = g.reset_index(drop=True)

            file_paths = (g["image_filename"]
                          .apply(lambda f: os.path.join(self.img_dir, f))
                          .tolist())
            labels = g[["predicted_steering_angle", "predicted_throttle"]].values

            T = len(file_paths)
            if T < self.seq_len:
                continue
            for start in range(0, T - self.seq_len + 1, self.seq_stride):
                end = start + self.seq_len
                seq_paths.append(file_paths[start:end])

                if self.label_from == "last":
                    lab = labels[end - 1]
                else:  # center
                    lab = labels[start + self.seq_len // 2]
                seq_labels.append(lab.astype(np.float32))

        return np.array(seq_paths), np.array(seq_labels, dtype=np.float32)

    @staticmethod
    def _decode_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        # Resize to (66, 200) to match PilotNet defaults
        img = tf.image.resize(img, [66, 200])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img

    def _process(self, paths, label):
        """
        paths: (seq_len,) tf.string
        label: (2,) tf.float32
        """
        imgs = tf.map_fn(
            fn=lambda p: self._decode_image(p),
            elems=paths,
            fn_output_signature=tf.float32,
            parallel_iterations=8
        )  # (T, H, W, C)
        return imgs, label

    def dataset(self):
        ds = tf.data.Dataset.from_tensor_slices((self.seq_paths, self.seq_labels))
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.seq_paths))
        ds = ds.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
