import os
import pandas as pd

IMG_EXTENSIONS = [
    '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_sequences(path):
    # Scenes (office, office2, carwelding, hospital)
    # Difficulty (Easy, Hard)
    # Sequences (P000, P001, ...)
    seqs = [os.path.join(path, scene, diff, seq) for scene in os.listdir(path) if os.path.isdir(os.path.join(path, scene)) for diff in os.listdir(os.path.join(path, scene)) if os.path.isdir(os.path.join(path, scene, diff)) for seq in os.listdir(os.path.join(path, scene, diff)) if os.path.isdir(os.path.join(path, scene, diff, seq)) ]

    return seqs

def read_tartanair(filepath):

    train_path = os.path.join(filepath, 'train')
    test_path = os.path.join(filepath, 'val')
    train_seqs = get_sequences(train_path)
    test_seqs = get_sequences(test_path)
    #train_seqs += test_seqs

    all_left_img = []
    all_right_img = []
    all_left_deps = []
    test_left_img = []
    test_right_img = []
    test_left_deps = []

    for seq in train_seqs:
        imgs = os.listdir(os.path.join(seq, 'cam0_on', 'data'))
        imgs.sort()
        for im in imgs:
            if is_image_file(im) and (os.path.exists(os.path.join(seq, 'cam1_on', 'data', im))) and (os.path.exists(os.path.join(seq, 'depth0', 'data', im))):
                all_left_img.append(seq + '/cam0_on/data/' + im)
                all_right_img.append(seq + '/cam1_on/data/' + im)
                all_left_deps.append(seq + '/depth0/data/' + im)

    for seq in test_seqs:
        imgs = os.listdir(os.path.join(seq, 'cam0_on', 'data'))
        imgs.sort()
        for im in imgs:
            if is_image_file(im) and (os.path.exists(os.path.join(seq, 'cam1_on', 'data', im))) and (os.path.exists(os.path.join(seq, 'depth0', 'data', im))):
                test_left_img.append(seq + '/cam0_on/data/' + im)
                test_right_img.append(seq + '/cam1_on/data/' + im)
                test_left_deps.append(seq + '/depth0/data/' + im)

    print("Number of training images: ", len(all_left_img))
    print("Number of testing images: ", len(test_left_img))
    return all_left_img, all_right_img, all_left_deps, test_left_img, test_right_img, test_left_deps
