import numpy as np
import cv2


def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def get_largest_incorrect_region(alpha, gt):

    largest_incorrect_BF = []
    for val in [0, 1]:

        incorrect = (gt == val) * (alpha != val)
        ret, labels_con = cv2.connectedComponents(incorrect.astype(np.uint8) * 255)
        label_unique, counts = np.unique(labels_con[labels_con != 0], return_counts=True)
        if(len(counts) > 0):
            largest_incorrect = labels_con == label_unique[np.argmax(counts)]
            largest_incorrect_BF.append(largest_incorrect)
        else:
            largest_incorrect_BF.append(np.zeros_like(incorrect))

    largest_incorrect_cat = np.argmax([np.count_nonzero(x) for x in largest_incorrect_BF])
    largest_incorrect = largest_incorrect_BF[largest_incorrect_cat]
    return largest_incorrect, largest_incorrect_cat


def robot_click(alpha, gt, trimap):
    incorrect_region, click_cat = get_largest_incorrect_region(alpha, gt)
    y, x = click_position(incorrect_region, trimap[:, :, click_cat])
    trimap[y, x, click_cat] = 1
    return trimap, incorrect_region, [y, x], click_cat


def click_position(largest_incorrect, clicks_cat):
    h, w = largest_incorrect.shape

    largest_incorrect_boundary = np.zeros((h + 2, w + 2))
    largest_incorrect_boundary[1:-1, 1:-1] = largest_incorrect
    clicks_cat_boundary = np.zeros((h + 2, w + 2))
    clicks_cat_boundary[1:-1, 1:-1] = clicks_cat

    uys, uxs = np.where(largest_incorrect_boundary > 0)

    if(uys.shape[0] == 0):
        return -1, -1

    no_click_mask = (1 - largest_incorrect_boundary)
    dist = dt(1 - no_click_mask)
    dist = dist[1:-1, 1:-1]
    y, x = np.unravel_index(dist.argmax(), dist.shape)

    return y, x


def jaccard(annotation, segmentation, void_pixels=None):
    # https://github.com/scaelles/DEXTR-PyTorch/blob/master/evaluation
    assert(annotation.shape == segmentation.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(annotation)
    assert(void_pixels.shape == annotation.shape)

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)
    if np.isclose(np.sum(annotation & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(segmentation & np.logical_not(void_pixels)), 0):
        return 1
    else:
        return np.sum(((annotation & segmentation) & np.logical_not(void_pixels))) / \
            np.sum(((annotation | segmentation) & np.logical_not(void_pixels)), dtype=np.float32)


def remove_non_fg_connected(alpha_np, fg_pos):

    if(np.count_nonzero(fg_pos) > 0):
        ys, xs = np.where(fg_pos == 1)

        alpha_np_bin = alpha_np > 0.5
        ret, labels_con = cv2.connectedComponents((alpha_np_bin * 255).astype(np.uint8))

        labels_f = []
        for y, x in zip(ys, xs):
            if(labels_con[y, x] != 0):
                labels_f.append(labels_con[y, x])
        fg_con = np.zeros_like(alpha_np)
        for lab in labels_f:
            fg_con[labels_con == lab] = 1

        alpha_np[fg_con == 0] = 0

    return alpha_np
