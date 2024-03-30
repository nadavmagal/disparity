import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
    # Load the left and right images in grayscale
    # dir_name = 'not_parallel1'
    imgL, imgR = get_images()

    dstL, dstR = rectify_images_uncalibrated_cameras(imgL, imgR)

    disparity_norm = calculate_disparity(dstL, dstR)

    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(dstL)
    axs[0,0].set_title("dstL")
    axs[0,1].imshow(dstR)
    axs[0,1].set_title("dstR")
    axs[1,0].imshow(disparity_norm)
    axs[1,0].set_title("disparity_norm")
    plt.show(block=False)

    return


def calculate_disparity(dstL, dstR):
    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=0, blockSize=51)
    # stereo = cv2.StereoSGBM(numDisparities=16, blockSize=15)
    # Compute the disparity image
    disparity = stereo.compute(dstL, dstR)
    # Normalize the disparity image for display
    norm_coeff = 255 / disparity.max()
    disparity_norm = disparity * norm_coeff / 255
    disparity_norm = disparity - np.min(disparity)
    disparity_norm = ((disparity_norm / np.max(disparity_norm)) * 255).astype(np.uint8)
    return disparity_norm


def rectify_images_uncalibrated_cameras(imgL, imgR):
    # rectify images
    sift = cv2.SIFT_create()
    ###find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)
    ###FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    ###ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    good = good[100:400]
    # Draw the matches
    img_matches = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Show the result
    plt.figure()
    plt.imshow(img_matches)
    plt.show(block=False)
    a = 3
    # Computation of the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # Obtainment of the rectification matrix and use of the warpPerspective to transform them...
    pts1 = pts1[:, :][mask.ravel() == 1]
    pts2 = pts2[:, :][mask.ravel() == 1]
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
    p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))
    retBool, rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew, p2fNew, F, imgL.shape[::-1])
    dstR = cv2.warpPerspective(imgL, rectmat1, imgL.shape[::-1])
    dstL = cv2.warpPerspective(imgR, rectmat2, imgL.shape[::-1])
    if True:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(imgL)
        axs[0, 0].set_title('imgL')
        axs[0, 1].imshow(imgR)
        axs[0, 1].set_title('imgr')
        axs[1, 0].imshow(dstR)
        axs[1, 0].set_title('dstR')
        axs[1, 1].imshow(dstL)
        axs[1, 1].set_title('dstL')
        plt.show(block=False)
    return dstL, dstR


def get_images():
    dir_name = 'parallel3'
    imgL = cv2.imread(dir_name + '/left.jpg', 0)
    imgR = cv2.imread(dir_name + '/right.jpg', 0)
    if imgL is None:
        imgL = cv2.imread(dir_name + '/left.png', 0)
        imgR = cv2.imread(dir_name + '/right.png', 0)
    return imgL, imgR


if __name__ == '__main__':
    main()

