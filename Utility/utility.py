"""
MIT License

Copyright (c) 2022 SLAMcore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
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
