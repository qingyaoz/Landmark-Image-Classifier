"""
Script to create an augmented dataset.
"""

import argparse
import csv
import glob
import os
import sys
import numpy as np
from scipy.ndimage import rotate
from imageio.v3 import imread, imwrite
from torchvision import transforms as tf

import rng_control


def Rotate(deg=20):
    """Return function to rotate image."""

    def _rotate(img):
        """
        Rotate a random integer amount in the range (-deg, deg) (inclusive).
        Keep the dimensions the same and fill any missing pixels with black.

        img: H x W x C numpy array
        returns: H x W x C numpy array
        """
        degree = np.random.randint(-deg, deg+1)
        rotated_img = rotate(img, angle=degree, axes=(0, 1), reshape=False, mode='constant', cval=0)
        rotated_img = rotated_img.astype(img.dtype)
        # axes=(0, 1) --> rotate in 2D
        # reshape=False --> Keep the dimensions the same
        # mode='constant', cval=0 --> use black to fill missing pixels
        # print(f"shape compare: {img.shape == rotated_img.shape}")
        return rotated_img # 保持图片数据的一致性和准确性

    return _rotate


def Grayscale():
    """Return function to grayscale image."""

    def _grayscale(img):
        """
        Return 3-channel grayscale of image.

        img: H x W x C numpy array
        returns: H x W x C numpy array
        """
        gray = np.mean(img, axis=-1).round().astype(np.uint8)
        gray_img = np.stack([gray, gray, gray], axis=-1)
        return gray_img

    return _grayscale


def augment(filename, transforms, n=1, original=True):
    """Augment image at filename.

    filename: name of image to be augmented
    transforms: List of image transformations
    n: number of augmented images to save
    original: whether to include the original images in the augmented dataset or not
    returns: a list of augmented images, where the first image is the original
    """
    print(f"Augmenting {filename}")
    img = imread(filename)
    res = [img] if original else []
    for i in range(n):
        new = img
        for transform in transforms:
            new = transform(new)
        res.append(new)
    return res


def main(args):
    """Create augmented dataset."""
    reader = csv.DictReader(open(args.input, "r"), delimiter=",")
    writer = csv.DictWriter(
        open(f"{args.datadir}/augmented_landmarks.csv", "w"),
        fieldnames=["filename", "semantic_label", "partition", "numeric_label", "task"],
    )
    augment_partitions = set(args.partitions)

    augmentations = [Grayscale()] # Rotate(), 

    writer.writeheader()
    os.makedirs(f"{args.datadir}/augmented/", exist_ok=True)
    for f in glob.glob(f"{args.datadir}/augmented/*"):
        print(f"Deleting {f}")
        os.remove(f)
    for row in reader:
        if row["partition"] not in augment_partitions:
            imwrite(
                f"{args.datadir}/augmented/{row['filename']}",
                imread(f"{args.datadir}/images/{row['filename']}"),
            )
            writer.writerow(row)
            continue
        imgs = augment(
            f"{args.datadir}/images/{row['filename']}",
            augmentations,
            n=1,
            original=False,  # change to False to exclude original image.
        )
        for i, img in enumerate(imgs):
            fname = f"{row['filename'][:-4]}_aug_{i}.png"
            imwrite(f"{args.datadir}/augmented/{fname}", img)
            writer.writerow(
                {
                    "filename": fname,
                    "semantic_label": row["semantic_label"],
                    "partition": row["partition"],
                    "numeric_label": row["numeric_label"],
                    "task": row["task"],
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("datadir", help="Data directory", default="./data/")
    parser.add_argument(
        "-p",
        "--partitions",
        nargs="+",
        help="Partitions (train|val|test|challenge|none)+ to apply augmentations to. Defaults to train",
        default=["train"],
    )
    main(parser.parse_args(sys.argv[1:]))
