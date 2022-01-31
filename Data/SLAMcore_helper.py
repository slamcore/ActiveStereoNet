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
import os
import pandas as pd

IMG_EXTENSIONS = [
    '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_slamcore(filepath):

    train_path = os.path.join(filepath, 'train')
    test_path = os.path.join(filepath, 'test')
    train_seqs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    test_seqs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]

    all_left_img = []
    all_right_img = []
    all_left_deps = []
    test_left_img = []
    test_right_img = []
    test_left_deps = []

    for seq in train_seqs:
        csv = pd.read_csv(os.path.join(train_path, seq, 'ir0', 'data.csv'))
        imgs = csv['filename'][csv['laser power'] > 50]
        for im in imgs:
            if is_image_file(im) and (os.path.exists(os.path.join(train_path, seq, 'ir1', 'data', im))) and (os.path.exists(os.path.join(train_path, seq, 'depth0', 'data', im))):
                all_left_img.append(train_path + '/' + seq + '/ir0/data/' + im)
                all_right_img.append(train_path + '/' + seq + '/ir1/data/' + im)
                all_left_deps.append(train_path + '/' + seq + '/depth0/data/' + im)

    for seq in test_seqs:
        csv = pd.read_csv(os.path.join(test_path, seq, 'ir0', 'data.csv'))
        imgs = csv['filename'][csv['laser power'] > 50]
        for im in imgs:
            if is_image_file(im) and (os.path.exists(os.path.join(test_path, seq, 'ir1', 'data', im))) and (os.path.exists(os.path.join(test_path, seq, 'depth0', 'data', im))):
                test_left_img.append(test_path + '/' + seq + '/ir0/data/' + im)
                test_right_img.append(test_path + '/' + seq + '/ir1/data/' + im)
                test_left_deps.append(test_path + '/' + seq + '/depth0/data/' + im)

    print("Number of training images: ", len(all_left_img))
    print("Number of testing images: ", len(test_left_img))
    return all_left_img, all_right_img, all_left_deps, test_left_img, test_right_img, test_left_deps
