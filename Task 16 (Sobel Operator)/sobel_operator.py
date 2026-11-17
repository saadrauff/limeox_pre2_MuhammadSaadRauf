
"""
TASK 16 - IMAGE PROCESSING
SOBEL OPERATOR - EDGE DETECTION

Dataset: CIFAR-10
Input: Color images
Operations:
    1. Load CIFAR-10 dataset
    2. Convert images to grayscale
    3. Apply Sobel X
    4. Apply Sobel Y
    5. Calculate gradient magnitude
    6. Save processed images
    7. Generate comparison visualizations
    8. Save processing summary

The original text classification dataset is not modified.
"""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.datasets import CIFAR10


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RESULT_DIR = BASE_DIR / "sobel_results"

ORIGINAL_DIR = RESULT_DIR / "original"
GRAYSCALE_DIR = RESULT_DIR / "grayscale"
SOBEL_X_DIR = RESULT_DIR / "sobel_x"
SOBEL_Y_DIR = RESULT_DIR / "sobel_y"
MAGNITUDE_DIR = RESULT_DIR / "sobel_magnitude"
COMPARISON_DIR = RESULT_DIR / "comparisons"

# Number of images to process.
# Set to None to process all 50,000 training images.
MAX_IMAGES = 500

# Sobel kernel size
KERNEL_SIZE = 3

# CIFAR-10 training dataset
USE_TRAIN_DATA = True

# Display settings
DISPLAY_SAMPLE_COUNT = 12


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

directories = [
    DATA_DIR,
    RESULT_DIR,
    ORIGINAL_DIR,
    GRAYSCALE_DIR,
    SOBEL_X_DIR,
    SOBEL_Y_DIR,
    MAGNITUDE_DIR,
    COMPARISON_DIR
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. SOBEL OPERATOR FUNCTION
# ============================================================

def apply_sobel(image):
    """
    Apply Sobel edge detection to an RGB image.

    Returns:
        grayscale
        sobel_x
        sobel_y
        magnitude
    """

    # Convert RGB image into grayscale
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Sobel X: horizontal intensity gradient
    gradient_x = cv2.Sobel(
        grayscale,
        cv2.CV_64F,
        1,
        0,
        ksize=KERNEL_SIZE
    )

    # Sobel Y: vertical intensity gradient
    gradient_y = cv2.Sobel(
        grayscale,
        cv2.CV_64F,
        0,
        1,
        ksize=KERNEL_SIZE
    )

    # Convert gradients into displayable images
    sobel_x = cv2.convertScaleAbs(gradient_x)
    sobel_y = cv2.convertScaleAbs(gradient_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    # Normalize magnitude between 0 and 255
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    magnitude = magnitude.astype(np.uint8)

    return grayscale, sobel_x, sobel_y, magnitude


# ============================================================
# 4. GENERATE COMPARISON VISUALIZATION
# ============================================================

def save_comparison(
    original,
    grayscale,
    sobel_x,
    sobel_y,
    magnitude,
    image_name
):

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    images = [
        original,
        grayscale,
        sobel_x,
        sobel_y,
        magnitude
    ]

    titles = [
        "Original Image",
        "Grayscale",
        "Sobel X",
        "Sobel Y",
        "Sobel Magnitude"
    ]

    for ax, image, title in zip(axes, images, titles):

        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    output_path = COMPARISON_DIR / f"{image_name}.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 5. LOAD CIFAR-10 DATASET
# ============================================================

def main():

    print("=" * 65)
    print("TASK 16 - SOBEL OPERATOR")
    print("IMAGE PROCESSING AND EDGE DETECTION")
    print("=" * 65)

    print("\nLoading CIFAR-10 dataset...")

    dataset = CIFAR10(
        root=str(DATA_DIR / "CIFAR-10"),
        train=USE_TRAIN_DATA,
        download=True
    )

    total_images = len(dataset)

    if MAX_IMAGES is None:
        images_to_process = total_images
    else:
        images_to_process = min(
            MAX_IMAGES,
            total_images
        )

    print(f"\nDataset: CIFAR-10")
    print(f"Total available images: {total_images}")
    print(f"Images to process: {images_to_process}")
    print(f"Image size: 32 x 32")
    print(f"Sobel kernel size: {KERNEL_SIZE}")

    print("\nStarting Sobel processing...\n")


    # ========================================================
    # 6. PROCESS IMAGES
    # ========================================================

    summary_path = RESULT_DIR / "summary.csv"

    processed_count = 0

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "Image_ID",
            "Class",
            "Width",
            "Height",
            "Channels",
            "Processing_Status"
        ])

        for index in range(images_to_process):

            try:

                # Load image and label
                pil_image, label = dataset[index]

                image = np.asarray(
                    pil_image.convert("RGB")
                )

                class_name = dataset.classes[label]

                image_name = f"{index:05d}_{class_name}"

                # Apply Sobel
                grayscale, sobel_x, sobel_y, magnitude = (
                    apply_sobel(image)
                )

                # Save original image
                cv2.imwrite(
                    str(ORIGINAL_DIR / f"{image_name}.png"),
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_RGB2BGR
                    )
                )

                # Save grayscale image
                cv2.imwrite(
                    str(GRAYSCALE_DIR / f"{image_name}.png"),
                    grayscale
                )

                # Save Sobel X
                cv2.imwrite(
                    str(SOBEL_X_DIR / f"{image_name}.png"),
                    sobel_x
                )

                # Save Sobel Y
                cv2.imwrite(
                    str(SOBEL_Y_DIR / f"{image_name}.png"),
                    sobel_y
                )

                # Save final Sobel magnitude
                cv2.imwrite(
                    str(MAGNITUDE_DIR / f"{image_name}.png"),
                    magnitude
                )

                # Save comparison for first 12 images
                if index < DISPLAY_SAMPLE_COUNT:

                    save_comparison(
                        image,
                        grayscale,
                        sobel_x,
                        sobel_y,
                        magnitude,
                        image_name
                    )

                writer.writerow([
                    index,
                    class_name,
                    image.shape[1],
                    image.shape[0],
                    image.shape[2],
                    "Success"
                ])

                processed_count += 1

                if (
                    (index + 1) % 50 == 0
                    or index + 1 == images_to_process
                ):

                    print(
                        f"Processed: {index + 1}/"
                        f"{images_to_process}"
                    )

            except Exception as error:

                print(
                    f"Error processing image {index}: {error}"
                )

                writer.writerow([
                    index,
                    "Unknown",
                    "",
                    "",
                    "",
                    f"Failed: {error}"
                ])


    # ========================================================
    # 7. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("SOBEL OPERATOR PROCESSING COMPLETED")
    print("=" * 65)

    print(f"\nTotal dataset images: {total_images}")
    print(f"Requested images: {images_to_process}")
    print(f"Successfully processed: {processed_count}")
    print(
        f"Failed images: "
        f"{images_to_process - processed_count}"
    )

    print("\nGenerated folders:")

    print(f"Original: {ORIGINAL_DIR}")
    print(f"Grayscale: {GRAYSCALE_DIR}")
    print(f"Sobel X: {SOBEL_X_DIR}")
    print(f"Sobel Y: {SOBEL_Y_DIR}")
    print(f"Magnitude: {MAGNITUDE_DIR}")
    print(f"Comparisons: {COMPARISON_DIR}")

    print(f"\nSummary CSV: {summary_path}")

    print("\nTask 16 completed successfully!")


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
