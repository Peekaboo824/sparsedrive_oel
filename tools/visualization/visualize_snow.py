import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

class AddSnowEffect:

    def __init__(self, snow_intensity=0.5):
        self.intensity = snow_intensity

    def _add_snow_to_single_image(self, img):
        if self.intensity < 0.01:
            return img
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image must be a np.ndarray, but got {type(img)}")
        original_dtype = img.dtype
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            img_for_cv2 = (img * 255).astype(np.uint8)
        else:
            img_for_cv2 = img.astype(np.uint8)
        height, width, _ = img_for_cv2.shape
        snow_layer = np.zeros_like(img_for_cv2, dtype=np.uint8)
        num_flakes = int(height * width * 0.05 * self.intensity)
        snow_x = np.random.randint(0, width, num_flakes)
        snow_y = np.random.randint(0, height, num_flakes)
        flake_sizes = np.random.randint(1, 4, num_flakes)
        flake_brightness = np.random.randint(150, 256, num_flakes)
        for i in range(num_flakes):
            brightness = int(flake_brightness[i])
            cv2.circle(snow_layer, (snow_x[i], snow_y[i]),
                       flake_sizes[i], (brightness, brightness, brightness),
                       thickness=-1)
        snow_layer = cv2.GaussianBlur(snow_layer, (7, 7), 0)
        snowy_img_uint8 = cv2.add(img_for_cv2, snow_layer)
        final_img = snowy_img_uint8.astype(original_dtype)
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            final_img /= 255.0
        return final_img


def main():
    print("downloading image...")
    image_url = ""  # add image url
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        original_image = np.array(pil_image)
    except requests.exceptions.RequestException as e:
        print(f"cannot download image: {e}")
        return

    print("applying snow effect...")
    heavy_snow_augmenter = AddSnowEffect(snow_intensity=0.1)

    snowy_image = heavy_snow_augmenter._add_snow_to_single_image(original_image)

    print("generating snow image...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(snowy_image)
    axes[1].set_title("Image with Heavy Snow Effect", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()