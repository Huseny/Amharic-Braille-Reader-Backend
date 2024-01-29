import PIL
import cv2
import numpy as np


MIN_RECTS = 10
MAX_ROTATION = 0.2
MIN_ROTATION = 0.02

def _center_of_char(ch):
    """
    :return: x,y - center of ch.refined_box
    """
    return (ch.refined_box[0] + ch.refined_box[2])/2, (ch.refined_box[1] + ch.refined_box[3])/2


def find_transformation(lines, img_size_wh):
    """
    Finds alignment transform
    :param lines:
    :return:
    """
    hom = None
    if len(lines) > 0:
        bounds = lines[0].chars[0].refined_box
        for ln in lines:
            for ch in ln.chars:
                bounds = [min(bounds[0], ch.refined_box[0]), min(bounds[1], ch.refined_box[1]),
                          max(bounds[2], ch.refined_box[2]), max(bounds[3], ch.refined_box[3])]
        sum_slip = 0
        sum_weights = 0
        for ln in lines:
            if len(ln.chars) > MIN_RECTS:
                pt1 = _center_of_char(ln.chars[len(ln.chars)//5])
                pt2 = _center_of_char(ln.chars[len(ln.chars)*4//5])
                dx = pt2[0] - pt1[0]
                slip = (pt2[1] - pt1[1])/dx
                sum_slip += slip * dx
                sum_weights += dx
        if sum_weights > 0:
            angle = sum_slip / sum_weights  # rad -> grad
            if abs(angle) < MAX_ROTATION and abs(angle) > MIN_ROTATION:
                scale = 1.
                center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
                hom = cv2.getRotationMatrix2D(center, angle * 59, scale)
                old_points = np.array([[(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[0], bounds[3]), (bounds[2], bounds[3])]])
                new_points = cv2.transform(old_points, hom)
                scale = min(
                    center[0]/(center[0]-new_points[0][0][0]), center[1]/(center[1]-new_points[0][0][1]),
                    (img_size_wh[0]-center[0]) / (new_points[0][1][0] - center[0]), center[1]/(center[1]-new_points[0][1][1]),
                    center[0] / (center[0] - new_points[0][2][0]), (img_size_wh[1]-center[1]) / (new_points[0][2][1] - center[1]),
                    (img_size_wh[0]-center[0]) / (new_points[0][3][0] - center[0]), (img_size_wh[1]-center[1]) / (new_points[0][3][1] - center[1])
                )
                if scale < 1:
                    hom = cv2.getRotationMatrix2D(center, angle * 59, scale)
    return hom


def transform_image(img, hom):
    """
    transforms img and refined_box'es and original_box'es for chars at lines using homography matrix found by
    find_transformation(lines)
    :param img: PIL image
    :param lines:
    :param hom:
    :return: img, lines transformed
    """
    if hom.shape[0] == 3:
        img_transform = cv2.warpPerspective
    else:
        img_transform = cv2.warpAffine
    img = img_transform(np.asarray(img), hom, img.size, flags=cv2.INTER_LINEAR )
    img = PIL.Image.fromarray(img)
    return img

def transform_lines(lines, hom):
    if hom.shape[0] == 3:
        pts_transform = cv2.perspectiveTransform
    else:
        pts_transform = cv2.transform
    old_centers = np.array([[_center_of_char(ch) for ln in lines for ch in ln.chars]])
    new_centers = pts_transform(old_centers, hom)
    shifts = (new_centers - old_centers)[0]
    i = 0
    for ln in lines:
        for ch in ln.chars:
            ch.refined_box[0] += shifts[i, 0]
            ch.refined_box[1] += shifts[i, 1]
            ch.refined_box[2] += shifts[i, 0]
            ch.refined_box[3] += shifts[i, 1]
            ch.original_box[0] += shifts[i, 0]
            ch.original_box[1] += shifts[i, 1]
            ch.original_box[2] += shifts[i, 0]
            ch.original_box[3] += shifts[i, 1]
            i += 1
    return lines

def transform_rects(rects, hom):
    if len(rects):
        if hom.shape[0] == 3:
            pts_transform = cv2.perspectiveTransform
        else:
            pts_transform = cv2.transform
        old_centers = np.array([[((r[2]+r[0])/2, (r[3]+r[1])/2) for r in rects]])
        new_centers = pts_transform(old_centers, hom)
        shifts = (new_centers - old_centers)[0].tolist()
        rects = [
            (x[0][0] + x[1][0], x[0][1] + x[1][1], x[0][2] + x[1][0], x[0][3] + x[1][1]) + tuple(x[0][4:])
            for x in zip(rects, shifts)
        ]
    return rects
