import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load the left and right images in grayscale
    dir_name = 'parallel4'

    imgL = cv2.imread(dir_name + '/left.jpg', 0)
    imgR = cv2.imread(dir_name + '/right.jpg', 0)

    if imgL is None:
        imgL = cv2.imread(dir_name + '/left.png', 0)
        imgR = cv2.imread(dir_name + '/right.png', 0)

    # Initialize the stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=0, blockSize=21) #best for image 4

    # win_size = 21
    # stereo = cv2.StereoSGBM(minDisparity=0,
    #                         numDisparities=21,
    #                         SADWindowSize=win_size,
    #                         uniquenessRatio=10,
    #                         speckleWindowSize=100,
    #                         speckleRange=32,
    #                         disp12MaxDiff=1,
    #                         P1=8 * 3 * win_size ** 2,
    #                         P2=32 * 3 * win_size ** 2,
    #                         fullDP=True
    #                         )
    # stereo = cv2.StereoSGBM(numDisparities=16, blockSize=15)

    # Compute the disparity image
    disparity = stereo.compute(imgL, imgR)

    # Normalize the disparity image for display
    norm_coeff = 255 / disparity.max()
    disparity_norm = disparity * norm_coeff / 255

    disparity_norm = disparity - np.min(disparity)
    disparity_norm = ((disparity_norm / np.max(disparity_norm)) * 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(imgL)
    axs[0, 0].set_title("imgL")
    axs[0, 1].imshow(imgR)
    axs[0, 1].set_title("imgR")
    axs[1, 0].imshow(disparity_norm)
    axs[1, 0].set_title("disparity_norm")
    plt.show(block=False)

    return


if __name__ == '__main__':
    main()
