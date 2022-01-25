import cv2
import numpy as np

def map_gray_to_colour(gray, max_value=256):
     remap = gray.astype('float') * 256 / max_value
     return cv2.applyColorMap(np.round(remap).astype('uint8'), cv2.COLORMAP_JET)

def put_text(img, text, location = 'bottom'):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2
    font_colour = (255, 255, 255)

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, font_size, font_thickness)[0]

    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    textY = textsize[1] + 10
    if location == 'top':
        textY = img.shape[0] - textY

    # add text centered on image
    return cv2.putText(img.copy(), text, (textX, textY), font, font_size, font_colour, font_thickness)
