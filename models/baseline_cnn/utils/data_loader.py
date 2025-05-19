import os
import tensorflow as tf

class DrivingDataset:
    """
    Builds a tf.data.Dataset for single-frame CNN training.
    Expects a pandas DataFrame with columns:
      - 'image_filename'
      - 'predicted_steering_angle' # predicted by the PID
      - 'predicted_throttle'
    """
    def __init__(self, df, img_dir, batch_size=64, shuffle=True):
        self.img_dir    = img_dir
        self.batch_size = batch_size
        self.shuffle    = shuffle

        # Precompute full paths and labels
        self.file_paths = [
            os.path.join(self.img_dir, fname)
            for fname in df['image_filename'].tolist()
        ]
        self.labels = df[['predicted_steering_angle', 'predicted_throttle']].values.tolist()

    def _process(self, file_path, label):
        # Load & preprocess image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [66, 200])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0

        # Convert labels to float32 tensor
        lbl = tf.cast(label, tf.float32)
        return img, lbl

    def dataset(self):
        ds = tf.data.Dataset.from_tensor_slices((self.file_paths, self.labels))
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.file_paths))
        ds = ds.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
