
"""
TASK 16 - SOBEL OPERATOR
KERNEL SIZE AND STRIDE COMPARISON

Dataset: CIFAR-10
Images: 500 training images

Kernel sizes: 3x3, 5x5, 7x7
Strides: 1, 2, 3

Operations:
    1. Load CIFAR-10 images
    2. Convert RGB images to grayscale
    3. Generate Sobel X and Sobel Y kernels
    4. Apply different kernel sizes
    5. Apply different strides
    6. Calculate gradient magnitude
    7. Compare before and after images
    8. Save all experiment results
    9. Generate kernel and stride comparisons
    10. Record processing statistics

Existing sobel_results folder is not modified.
"""

import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from pathlib import Path
from torchvision.datasets import CIFAR10


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "CIFAR-10"

RESULT_DIR = BASE_DIR / "sobel_kernel_stride_results"

KERNEL_SIZES = [3, 5, 7]

STRIDES = [1, 2, 3]

MAX_IMAGES = 500

DISPLAY_SAMPLES = 12

DEVICE = torch.device("cpu")

# CPU threads
torch.set_num_threads(6)


# ============================================================
# 2. CREATE RESULT DIRECTORIES
# ============================================================

RESULT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_DIR = RESULT_DIR / "comparisons"
KERNEL_COMPARISON_DIR = RESULT_DIR / "kernel_comparison"
STRIDE_COMPARISON_DIR = RESULT_DIR / "stride_comparison"

for directory in [
    COMPARISON_DIR,
    KERNEL_COMPARISON_DIR,
    STRIDE_COMPARISON_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. GENERATE SOBEL KERNELS
# ============================================================

def generate_sobel_kernels(kernel_size):
    """
    Generate separable Sobel X and Y kernels.

    Supports:
        3x3
        5x5
        7x7
    """

    if kernel_size not in [3, 5, 7]:
        raise ValueError("Supported kernels are 3, 5 and 7.")

    # Binomial smoothing coefficients
    smoothing = np.array(
        [
            1
        ]
    )

    for _ in range(kernel_size - 1):
        smoothing = np.convolve(
            smoothing,
            [1, 1]
        )

    # Derivative kernel
    smaller_smoothing = np.array([1])

    for _ in range(kernel_size - 3):
        smaller_smoothing = np.convolve(
            smaller_smoothing,
            [1, 1]
        )

    derivative = np.convolve(
        smaller_smoothing,
        [-1, 0, 1]
    )

    # Normalize smoothing
    smoothing = smoothing / smoothing.sum()

    # Normalize derivative
    derivative = derivative / np.abs(derivative).sum()

    # Create 2D kernels
    kernel_x = np.outer(
        smoothing,
        derivative
    )

    kernel_y = np.outer(
        derivative,
        smoothing
    )

    kernel_x = torch.tensor(
        kernel_x,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0).unsqueeze(0)

    kernel_y = torch.tensor(
        kernel_y,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0).unsqueeze(0)

    return kernel_x, kernel_y


# ============================================================
# 4. APPLY SOBEL WITH CUSTOM STRIDE
# ============================================================

def apply_sobel(image, kernel_size, stride):

    # RGB to grayscale
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Convert grayscale to PyTorch tensor
    image_tensor = torch.from_numpy(
        grayscale.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Generate Sobel kernels
    kernel_x, kernel_y = generate_sobel_kernels(
        kernel_size
    )

    # Apply convolution with selected stride
    gradient_x = F.conv2d(
        image_tensor,
        kernel_x,
        stride=stride
    )

    gradient_y = F.conv2d(
        image_tensor,
        kernel_y,
        stride=stride
    )

    # Calculate magnitude
    magnitude = torch.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    # Convert tensors to NumPy
    sobel_x = gradient_x.squeeze().cpu().numpy()
    sobel_y = gradient_y.squeeze().cpu().numpy()
    magnitude = magnitude.squeeze().cpu().numpy()

    # Normalize images for display
    def normalize_image(array):

        if array.ndim == 0:
            array = array.reshape(1, 1)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        array = np.abs(array)

        maximum = array.max()

        if maximum > 0:
            array = (array / maximum) * 255

        return array.astype(np.uint8)

    sobel_x = normalize_image(sobel_x)
    sobel_y = normalize_image(sobel_y)
    magnitude = normalize_image(magnitude)

    return grayscale, sobel_x, sobel_y, magnitude


# ============================================================
# 5. SAVE BEFORE AND AFTER COMPARISON
# ============================================================

def save_before_after(
    original,
    grayscale,
    sobel_x,
    sobel_y,
    magnitude,
    output_path,
    kernel_size,
    stride
):

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4)
    )

    images = [
        original,
        grayscale,
        sobel_x,
        sobel_y,
        magnitude
    ]

    titles = [
        "Before: Original",
        "Grayscale",
        "Sobel X",
        "Sobel Y",
        "After: Magnitude"
    ]

    for ax, image, title in zip(
        axes,
        images,
        titles
    ):

        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

        ax.set_title(title)

        ax.axis("off")

    fig.suptitle(
        f"Kernel: {kernel_size}x{kernel_size} | Stride: {stride}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 6. SAVE KERNEL COMPARISON
# ============================================================

def save_kernel_comparison(
    original,
    results,
    image_id
):

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    for index, kernel_size in enumerate(
        KERNEL_SIZES,
        start=1
    ):

        magnitude = results[kernel_size]

        axes[index].imshow(
            magnitude,
            cmap="gray"
        )

        axes[index].set_title(
            f"Kernel {kernel_size}x{kernel_size}"
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        KERNEL_COMPARISON_DIR / f"image_{image_id:05d}.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# 7. SAVE STRIDE COMPARISON
# ============================================================

def save_stride_comparison(
    original,
    results,
    image_id,
    kernel_size
):

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    for index, stride in enumerate(
        STRIDES,
        start=1
    ):

        magnitude = results[stride]

        axes[index].imshow(
            magnitude,
            cmap="gray"
        )

        axes[index].set_title(
            f"Stride {stride}"
        )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Kernel Size: {kernel_size}x{kernel_size}"
    )

    plt.tight_layout()

    plt.savefig(
        STRIDE_COMPARISON_DIR /
        f"kernel_{kernel_size}_image_{image_id:05d}.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("TASK 16 - SOBEL KERNEL AND STRIDE COMPARISON")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Kernel sizes: {KERNEL_SIZES}")
    print(f"Strides: {STRIDES}")
    print(f"Images: {MAX_IMAGES}")

    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")

    dataset = CIFAR10(
        root=str(DATA_DIR.parent),
        train=True,
        download=True
    )

    image_count = min(
        MAX_IMAGES,
        len(dataset)
    )

    print(f"Available images: {len(dataset)}")
    print(f"Processing images: {image_count}")

    # Create experiment folders
    experiment_dirs = {}

    for kernel_size in KERNEL_SIZES:

        for stride in STRIDES:

            experiment_name = (
                f"kernel_{kernel_size}_stride_{stride}"
            )

            experiment_dir = RESULT_DIR / experiment_name

            for subfolder in [
                "sobel_x",
                "sobel_y",
                "magnitude",
                "comparisons"
            ]:

                (experiment_dir / subfolder).mkdir(
                    parents=True,
                    exist_ok=True
                )

            experiment_dirs[
                (kernel_size, stride)
            ] = experiment_dir

    # Statistics
    statistics = {
        (k, s): {
            "times": [],
            "width": None,
            "height": None
        }
        for k in KERNEL_SIZES
        for s in STRIDES
    }

    summary_path = RESULT_DIR / "summary.csv"

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for image_id in range(image_count):

        pil_image, label = dataset[image_id]

        original = np.asarray(
            pil_image.convert("RGB")
        )

        class_name = dataset.classes[label]

        print(
            f"\nProcessing image {image_id + 1}/{image_count} "
            f"({class_name})"
        )

        kernel_results = {}

        # ----------------------------------------------------
        # Kernel and stride combinations
        # ----------------------------------------------------

        for kernel_size in KERNEL_SIZES:

            stride_results = {}

            for stride in STRIDES:

                output_dir = experiment_dirs[
                    (kernel_size, stride)
                ]

                start_time = time.perf_counter()

                grayscale, sobel_x, sobel_y, magnitude = (
                    apply_sobel(
                        original,
                        kernel_size,
                        stride
                    )
                )

                processing_time = (
                    time.perf_counter() - start_time
                )

                stats = statistics[
                    (kernel_size, stride)
                ]

                stats["times"].append(
                    processing_time
                )

                stats["height"], stats["width"] = (
                    magnitude.shape
                )

                image_name = f"{image_id:05d}_{class_name}"

                # Save Sobel X
                cv2.imwrite(
                    str(
                        output_dir / "sobel_x" /
                        f"{image_name}.png"
                    ),
                    sobel_x
                )

                # Save Sobel Y
                cv2.imwrite(
                    str(
                        output_dir / "sobel_y" /
                        f"{image_name}.png"
                    ),
                    sobel_y
                )

                # Save magnitude
                cv2.imwrite(
                    str(
                        output_dir / "magnitude" /
                        f"{image_name}.png"
                    ),
                    magnitude
                )

                # Before and after comparisons
                if image_id < DISPLAY_SAMPLES:

                    save_before_after(
                        original,
                        grayscale,
                        sobel_x,
                        sobel_y,
                        magnitude,
                        output_dir / "comparisons" /
                        f"{image_name}.png",
                        kernel_size,
                        stride
                    )

                stride_results[stride] = magnitude

            kernel_results[kernel_size] = (
                stride_results[1]
            )

            # Save stride comparison for this kernel
            if image_id < DISPLAY_SAMPLES:

                save_stride_comparison(
                    original,
                    stride_results,
                    image_id,
                    kernel_size
                )

        # Save kernel comparison using stride 1
        if image_id < DISPLAY_SAMPLES:

            save_kernel_comparison(
                original,
                kernel_results,
                image_id
            )

        if (
            (image_id + 1) % 50 == 0
            or image_id + 1 == image_count
        ):

            print(
                f"Completed: {image_id + 1}/{image_count}"
            )

    # ========================================================
    # 9. SAVE SUMMARY CSV
    # ========================================================

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "Kernel_Size",
            "Stride",
            "Images_Processed",
            "Output_Width",
            "Output_Height",
            "Mean_Processing_Time_Seconds",
            "Total_Processing_Time_Seconds"
        ])

        for kernel_size in KERNEL_SIZES:

            for stride in STRIDES:

                stats = statistics[
                    (kernel_size, stride)
                ]

                writer.writerow([
                    f"{kernel_size}x{kernel_size}",
                    stride,
                    len(stats["times"]),
                    stats["width"],
                    stats["height"],
                    f"{np.mean(stats['times']):.6f}",
                    f"{np.sum(stats['times']):.6f}"
                ])

    total_time = time.perf_counter() - total_start

    # ========================================================
    # 10. FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("KERNEL AND STRIDE EXPERIMENT COMPLETED")
    print("=" * 70)

    print(f"\nImages processed: {image_count}")
    print(f"Kernel sizes tested: {KERNEL_SIZES}")
    print(f"Strides tested: {STRIDES}")

    print("\nOutput dimensions:")

    for kernel_size in KERNEL_SIZES:

        for stride in STRIDES:

            stats = statistics[
                (kernel_size, stride)
            ]

            print(
                f"Kernel {kernel_size}x{kernel_size} | "
                f"Stride {stride} | "
                f"Output {stats['width']}x{stats['height']}"
            )

    print(f"\nTotal experiment time: {total_time:.2f} seconds")

    print(f"\nResults folder: {RESULT_DIR}")

    print(f"Summary CSV: {summary_path}")

    print("\nAll experiments completed successfully!")


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    main()
