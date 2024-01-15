import numpy as np
from PIL import Image
import random


class ImageProcessor:
    
    def __init__(self, target_size=None, resample_filter=None, color_space=None, random_seed=None):
        """
        random_seed should be unique per image. It is best to use the image id.
        """
        if target_size is None:
            target_size = (230,230)
        self.target_size = target_size

        if resample_filter is None:
            resample_filter = Image.Resampling.BOX
        self.resample_filter = resample_filter

        if color_space is None:
            color_space = "RGB"
        self.color_space = color_space

        if random_seed is None:
            random_seed = False
        self.random_seed = random_seed

    def PrepareImage(self, img):
        img = self.ChangeColorSpace(img)
        img = self.Rescale(img)
        return img

    def Rescale(self, img):
        img = img.resize(size=self.target_size, resample=self.resample_filter)
        return img

    def ChangeColorSpace(self, img):
        color_space = str.upper(self.color_space)
        try:
            img = img.convert(mode=color_space)
        except ValueError as e:
            print(f"Invalid color space or cannot convert to selected color space. Defaulting to RGB.\nERROR: {e}")
            img = img.convert(mode="RGB")
        finally:
            return img

    # Random effect (Mirror / Warp / Blur)
    def RandomEffect(self, img, effect):
        random.seed(self.random_seed)
        if effect is None:
                effect = random.choice(["Rotate", "Flip"])

        if "Rotate" in effect:
            img = img.rotate(random.randint(0,359))
        if "Flip" in effect:
            flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
            img = img.transpose(flip)

        return img

    