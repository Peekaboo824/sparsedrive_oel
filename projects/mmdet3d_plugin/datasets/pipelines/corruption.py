import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class AddRainEffect:
    """
    Adds a rain effect to one or more image views
    """

    def __init__(self, rain_intensity=0.4, rain_angle=-20, rain_drop_length=20):
        """
        Args:
            rain_intensity (float): The intensity of the rain, recommended between 0.1 and 1.0.
                                    This controls the transparency of the raindrops.
            rain_angle (int): The slant angle of the raindrops.
            rain_drop_length (int): The length of the raindrops.
        """
        self.intensity = rain_intensity
        self.angle = rain_angle
        self.drop_length = rain_drop_length

    def _add_rain_to_single_image(self, img):
        """Applies rain effect to a single image."""
        if self.intensity < 0.01:
            return img

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image must be a np.ndarray, but got {type(img)}")

        # Save the original data type and prepare a uint8 image for OpenCV operations.
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

        # Convert the image back to its original data type and range.
        final_img = rainy_img_uint8.astype(original_dtype)
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            final_img /= 255.0

        return final_img

    def __call__(self, results):
        """Main call function of the pipeline."""
        processed_imgs = []
        for img in results['img']:
            processed_imgs.append(self._add_rain_to_single_image(img))

        results['img'] = processed_imgs
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(intensity={self.intensity}, angle={self.angle}, drop_length={self.drop_length})'



@PIPELINES.register_module()
class AddFogEffect:
    """
    Adds a fog effect to one or more image views.
    """
    def __init__(self, fog_intensity=0.5, fog_color=(220, 220, 220)):
        """
        Args:
            fog_intensity (float): The density of the fog, recommended between 0.1 and 1.0.
                                   A larger value means denser fog.
            fog_color (tuple): The color of the fog, as an (R, G, B) tuple.
        """
        self.intensity = fog_intensity
        self.color = fog_color

    def _add_fog_to_single_image(self, img):
        """Applies fog effect to a single image."""
        if self.intensity < 0.01:
            return img

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image must be a np.ndarray, but got {type(img)}")

        # Save the original data type and prepare a uint8 image for OpenCV operations.
        original_dtype = img.dtype
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            img_for_cv2 = (img * 255).astype(np.uint8)
        else:
            img_for_cv2 = img.astype(np.uint8)

        height, width, _ = img_for_cv2.shape

        # Create a solid color fog layer.
        fog_layer = np.full((height, width, 3), self.color, dtype=np.uint8)

        # Use addWeighted to blend the original image and the fog layer.
        # The weight of the original image is (1 - intensity), and the weight of the fog layer is intensity.
        foggy_img_uint8 = cv2.addWeighted(img_for_cv2, 1 - self.intensity, fog_layer, self.intensity, 0)

        # Convert the image back to its original data type and range.
        final_img = foggy_img_uint8.astype(original_dtype)
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            final_img /= 255.0

        return final_img

    def __call__(self, results):
        """Main call function of the pipeline."""
        processed_imgs = []
        for img in results['img']:
            processed_imgs.append(self._add_fog_to_single_image(img))

        results['img'] = processed_imgs
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(intensity={self.intensity}, color={self.color})'


@PIPELINES.register_module()
class AddSnowEffect:
    """
    Adds a snow effect to one or more image views.
    """
    def __init__(self, snow_intensity=0.5):
        """
        Args:
            snow_intensity (float): The intensity of the snow, recommended between 0.1 and 1.0.
                                    A larger value means heavier snow.
        """
        self.intensity = snow_intensity

    def _add_snow_to_single_image(self, img):
        """Applies snow effect to a single image."""
        if self.intensity < 0.01:
            return img

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image must be a np.ndarray, but got {type(img)}")

        # Save the original data type and prepare a uint8 image for OpenCV operations.
        original_dtype = img.dtype
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            img_for_cv2 = (img * 255).astype(np.uint8)
        else:
            img_for_cv2 = img.astype(np.uint8)

        height, width, _ = img_for_cv2.shape

        # Create a black layer for snowflakes.
        snow_layer = np.zeros_like(img_for_cv2, dtype=np.uint8)

        # Calculate the number of snowflakes based on intensity.
        num_flakes = int(height * width * 0.05 * self.intensity)

        # Randomly generate snowflake positions.
        snow_x = np.random.randint(0, width, num_flakes)
        snow_y = np.random.randint(0, height, num_flakes)

        # Randomly generate snowflake sizes and brightness.
        flake_sizes = np.random.randint(1, 4, num_flakes)
        flake_brightness = np.random.randint(150, 256, num_flakes)

        # Draw all snowflakes.
        for i in range(num_flakes):
            brightness = int(flake_brightness[i])
            cv2.circle(snow_layer, (snow_x[i], snow_y[i]),
                       flake_sizes[i], (brightness, brightness, brightness),
                       thickness=-1) # -1 means to fill the circle

        # Apply motion blur to the snow layer to simulate a falling effect.
        snow_layer = cv2.GaussianBlur(snow_layer, (7, 7), 0)

        # Overlay the snow layer onto the original image using a simple "Add" blend mode.
        # To avoid brightness overflow, we use cv2.add, which automatically handles saturation (clipping).
        snowy_img_uint8 = cv2.add(img_for_cv2, snow_layer)

        # Convert the image back to its original data type and range.
        final_img = snowy_img_uint8.astype(original_dtype)
        if np.max(img) <= 1.0 and original_dtype == np.float32:
            final_img /= 255.0

        return final_img

    def __call__(self, results):
        """Main call function of the pipeline."""
        processed_imgs = []
        for img in results['img']:
            processed_imgs.append(self._add_snow_to_single_image(img))

        results['img'] = processed_imgs
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(intensity={self.intensity})'