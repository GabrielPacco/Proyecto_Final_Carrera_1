import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

def extract_color_features(image):
    if image.dtype != np.uint8 and image.dtype != np.float32:
        image = image.astype(np.uint8)
    
    color_features = []
    if len(image.shape) == 2:
        channel_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        color_features.extend(channel_hist.flatten())
    else:
        for i in range(3):
            channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            color_features.extend(channel_hist.flatten())
    
    color_features = np.nan_to_num(color_features)
    return color_features

def extract_texture_features(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image

    glcm = graycomatrix((gray_image * 255).astype('uint8'), distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    texture_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        texture_props = graycoprops(glcm, prop)
        texture_features.extend(texture_props.flatten())

    texture_features = np.nan_to_num(texture_features)
    return texture_features

def extract_shape_features(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)

    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    perimeter = 0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

    shape_features = [np.nan_to_num(area), np.nan_to_num(perimeter)]
    return shape_features

def extract_all_features(segmented_image, original_image):
    color_feats = extract_color_features(original_image)
    texture_feats = extract_texture_features(segmented_image)
    shape_feats = extract_shape_features(segmented_image)

    color_feats = np.array(color_feats)
    texture_feats = np.array(texture_feats)
    shape_feats = np.array(shape_feats)

    all_features = np.concatenate((color_feats, texture_feats, shape_feats))
    return all_features
