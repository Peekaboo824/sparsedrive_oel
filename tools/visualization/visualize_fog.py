import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO


class AddFogEffect:

    def __init__(self, fog_intensity=0.5, fog_color=(220, 220, 220)):
        self.intensity = fog_intensity
        self.color = fog_color

    def _add_fog_to_single_image(self, img):
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
        fog_layer = np.full((height, width, 3), self.color, dtype=np.uint8)
        foggy_img_uint8 = cv2.addWeighted(img_for_cv2, 1 - self.intensity, fog_layer, self.intensity, 0)
        final_img = foggy_img_uint8.astype(original_dtype)
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            final_img /= 255.0
        return final_img

def main():
    print("downloading...")
    image_url = "" # add example image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        original_image = np.array(pil_image)
    except requests.exceptions.RequestException as e:
        print(f"cannot download image: {e}")
        return

    print("applying fog effect...")
    heavy_fog_augmenter = AddFogEffect(fog_intensity=0.5)

    foggy_image = heavy_fog_augmenter._add_fog_to_single_image(original_image)

    print("generating...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(foggy_image)
    axes[1].set_title("Image with Heavy Fog Effect", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()