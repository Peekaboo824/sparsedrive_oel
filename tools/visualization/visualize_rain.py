import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO


class AddRainEffect:

    def __init__(self, rain_intensity=0.5, rain_angle=-20, rain_drop_length=25, rain_drop_thickness=1):
        """
        Args:
            rain_intensity (float)
            rain_angle (int)
            rain_drop_length (int)
            rain_drop_thickness (int)
        """
        self.intensity = rain_intensity
        self.angle = rain_angle
        self.drop_length = rain_drop_length
        self.thickness = rain_drop_thickness

    def _add_rain_to_single_image(self, img):
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
        rain_layer = np.zeros_like(img_for_cv2, dtype=np.uint8)
        num_drops = int(height * width * 0.01 * self.intensity)

        starts_x = np.random.randint(0, width, num_drops)
        starts_y = np.random.randint(0, height, num_drops)

        rad_angle = np.deg2rad(self.angle)
        ends_x = (starts_x + self.drop_length * np.sin(rad_angle)).astype(int)
        ends_y = (starts_y + self.drop_length * np.cos(rad_angle)).astype(int)

        for i in range(num_drops):
            cv2.line(rain_layer, (starts_x[i], starts_y[i]), (ends_x[i], ends_y[i]),
                     (200, 200, 200), thickness=1, lineType=cv2.LINE_AA)

        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

        rainy_img_uint8 = cv2.addWeighted(img_for_cv2, 1.0, rain_layer, self.intensity, 0)

        final_img = rainy_img_uint8.astype(original_dtype)
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            final_img /= 255.0

        return final_img


def main():

    print("downloading image")
    image_url = "" # add image url
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        original_image = np.array(pil_image)
    except requests.exceptions.RequestException as e:
        print(f"cannot download image: {e}")
        return

    print("applying rain effect...")

    heavy_rain_augmenter = AddRainEffect(
        rain_intensity=0.5,
        rain_angle=-15,
        rain_drop_length=25,
        rain_drop_thickness=1
    )

    rainy_image = heavy_rain_augmenter._add_rain_to_single_image(original_image)

    print("generating rain image...")
    #
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(rainy_image)
    axes[1].set_title("Image with Heavy Rain Effect", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()