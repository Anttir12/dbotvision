import uuid
import os
from PIL import Image


def get_cropped_image(path: str):
    im = Image.open(path)
    crop = (2363, 46, 2426, 90)
    cropped_image = im.crop(crop)
    filename = str(uuid.uuid4())
    cropped_image.save(f'ow_icons/{filename}.png')


files = os.listdir('C:/Users/anntt/Documents/Overwatch/ScreenShots/Overwatch')
for f in files:
    get_cropped_image(f'C:/Users/anntt/Documents/Overwatch/ScreenShots/Overwatch/{f}')
