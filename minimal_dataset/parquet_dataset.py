"""
Parquet-backed image Dataset.

Optimized image pipeline:
    Parquet row
        -> encoded JPEG bytes
        -> torchvision decode_jpeg
        -> resize while still uint8
        -> return uint8 tensor

Float conversion and normalization are intentionally deferred to
DataLoader collation so they operate on a whole batch instead of
individually on every full-resolution image.
"""

from typing import Optional, Callable

import torch
import torch.nn.functional as F
import torchvision.io as tvio
import pyarrow.parquet as pq


class ParquetDataset:

    def __init__(
        self,
        parquet_path: str,
        max_samples: int = None,
        transform: Optional[Callable] = None,
        image_size=(64, 64),
    ):

        self.parquet_path = parquet_path
        self.transform = transform
        self.image_size = image_size

        # NOTE:
        # This is still eager.
        #
        # The current benchmark Parquet contains one very large row group,
        # so efficient random lazy access requires changing the Parquet
        # layout itself. This remains a separate architectural issue.
        self._table = pq.read_table(
            parquet_path
        )

        self._length = len(
            self._table
        )

        if max_samples is not None:
            self._length = min(
                self._length,
                max_samples
            )


    def __len__(self):
        return self._length


    def __getitem__(
        self,
        index: int
    ):

        if (
            index < 0
            or index >= self._length
        ):
            raise IndexError(
                f"Index {index} out of range "
                f"for dataset of size {self._length}"
            )


        # ------------------------------------------------------------
        # 1. Extract encoded JPEG and label
        # ------------------------------------------------------------

        row = self._table.slice(
            index,
            1
        ).to_pylist()[0]

        image_bytes = row["image"]
        label = row["label"]


        # ------------------------------------------------------------
        # 2. Python bytes -> uint8 encoded tensor
        # ------------------------------------------------------------

        encoded = torch.frombuffer(
            bytearray(image_bytes),
            dtype=torch.uint8
        )


        # ------------------------------------------------------------
        # 3. JPEG decode
        # ------------------------------------------------------------

        image = tvio.decode_jpeg(
            encoded,
            mode=tvio.ImageReadMode.RGB
        )


        # ------------------------------------------------------------
        # 4. Resize while still uint8
        #
        # Experiments showed that converting the original-resolution
        # image to float BEFORE resize is significantly more expensive.
        # ------------------------------------------------------------

        if (
            image.shape[-2:]
            != tuple(self.image_size)
        ):

            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)


        # ------------------------------------------------------------
        # 5. Optional transform
        #
        # Custom transforms may expect normalized float tensors.
        # Therefore the optimized uint8 path is used when no transform
        # is supplied. With a transform, convert before applying it.
        # ------------------------------------------------------------

        if self.transform is not None:

            image = image.float()
            image.div_(255.0)

            image = self.transform(
                image
            )


        return image, label
