"""Image processing utilities for tag detection, enhancement, blur detection, and variants."""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Provides computer vision and preprocessing operations on product tag images."""

    @staticmethod
    def detect_blur(image: np.ndarray) -> Tuple[bool, float]:
        """Detect whether the image is blurry using the variance of the Laplacian.

        Returns (is_blurry, variance).
        """
        if image is None or image.size == 0:
            return True, 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = variance < settings.blur_threshold
        logger.debug("Blur detection variance: %.2f (blurry=%s)", variance, is_blurry)
        return is_blurry, variance

    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast (CLAHE), deskew, and upscale small images."""
        if image is None or image.size == 0:
            return image

        enhanced = image.copy()

        # CLAHE contrast enhancement
        if len(enhanced.shape) == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)

        # Upscale if small
        h, w = enhanced.shape[:2]
        max_dim = max(h, w)
        if max_dim < 1200:
            scale = 1200.0 / max_dim
            enhanced = cv2.resize(
                enhanced,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        return enhanced

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order coordinates: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def detect_and_crop_tag(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Detect the largest bright, roughly rectangular contour, apply perspective transform.

        Falls back to original image if no quadrilateral tag is confidently found.
        """
        if image is None or image.size == 0:
            return image, False

        h, w = image.shape[:2]
        img_area = h * w

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian blur + Bilateral filter to smooth texture while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Canny edge detection & thresholding
        edged = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image, False

        # Filter contours by size (at least 5% of total image area, max 98%)
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 0.05 * img_area < area < 0.98 * img_area:
                valid_contours.append((area, c))

        valid_contours.sort(key=lambda x: x[0], reverse=True)

        for _, contour in valid_contours:
            peri = cv2.arcLength(contour, True)
            # Try varying epsilon approximations
            for eps_factor in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    rect = ImageProcessor._order_points(pts)
                    (tl, tr, br, bl) = rect

                    # Compute widths and heights
                    width_a = np.linalg.norm(br - bl)
                    width_b = np.linalg.norm(tr - tl)
                    max_width = max(int(width_a), int(width_b))

                    height_a = np.linalg.norm(tr - br)
                    height_b = np.linalg.norm(tl - bl)
                    max_height = max(int(height_a), int(height_b))

                    if max_width < 100 or max_height < 100:
                        continue

                    dst = np.array(
                        [
                            [0, 0],
                            [max_width - 1, 0],
                            [max_width - 1, max_height - 1],
                            [0, max_height - 1],
                        ],
                        dtype="float32",
                    )

                    matrix = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

                    # Ensure upright aspect ratio if tag is expected vertical
                    wh, ww = warped.shape[:2]
                    # If width is substantially larger than height, product tags are usually vertical
                    if ww > wh * 1.5:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                    logger.info("Successfully detected and cropped tag: %dx%d", max_width, max_height)
                    return warped, True

        logger.info("No 4-point tag contour detected, returning original image.")
        return image, False

    @staticmethod
    def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by 90, 180, or 270 degrees clockwise."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    @staticmethod
    def get_preprocessing_variants(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate diverse preprocessing variants to optimize barcode and OCR detection."""
        variants: List[Tuple[str, np.ndarray]] = []
        if image is None or image.size == 0:
            return variants

        variants.append(("original", image))

        # Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        variants.append(("grayscale", gray))

        # Upscaled 2x
        h, w = gray.shape[:2]
        upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled_2x", upscaled))

        # Sharpened
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, kernel)
        variants.append(("sharpened", sharpened))

        # Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        variants.append(("adaptive_threshold", thresh))

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(("clahe", clahe_img))

        # Rotations
        variants.append(("rotation_90", cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)))
        variants.append(("rotation_180", cv2.rotate(gray, cv2.ROTATE_180)))
        variants.append(("rotation_270", cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)))

        return variants

