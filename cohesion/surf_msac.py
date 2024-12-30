import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
## 使用SURF+MSAC+加权平均融合方法实现图像拼接
## 应用于后续的科研（数据集制作）
## 陈沛森 湖北汽车工业学院 13346895166@163.com


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, imgs, blending_mode="linearBlending", ratio=0.7):
        '''
            The main method to stitch image
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        print("Left img size (", hl, "*", wl, ")")
        print("Right img size (", hr, "*", wr, ")")

        # Step1 - extract the keypoints and features by SURF detector and descriptor
        print("Step1 - Extract the keypoints and features by SURFdetector and descriptor...")
        kps_l, features_l = self.detectAndDescribe(img_left)
        kps_r, features_r = self.detectAndDescribe(img_right)

        # Step2 - extract the match point with threshold (David Lowe’s ratio test)
        print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
        matches_pos = self.matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio)
        print("The number of matching points:", len(matches_pos))

        # Step3 - fit the homography model with RANSAC algorithm
        print("Step3 - Fit the best homography model with MSAC algorithm...")
        HomoMat = self.fitHomoMat(matches_pos)

        # Step4 - Warp image to create panoramic image
        print("Step4 - Warp image to create panoramic image...")
        warp_img = self.warp([img_left, img_right], HomoMat, blending_mode)

        return warp_img

    def detectAndDescribe(self, img):
        '''
        The Detector and Descriptor
        '''
        # SURF detector and descriptor
        surf =  cv2.xfeatures2d.SURF_create()
        kps, features = surf.detectAndCompute(img, None)

        return kps, features

    def matchKeyPoint(self, kps_l, kps_r, features_l, features_r, ratio):
        '''
            Match the Keypoints between two images using Brute-Force Matching.
        '''
        # Create a Brute-Force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Perform knnMatch to get the two nearest neighbors
        matches = bf.knnMatch(features_l, features_r, k=2)

        goodMatches_pos = []
        for m, n in matches:
            # Apply Lowe's ratio test
            if m.distance < ratio * n.distance:
                psA = (int(kps_l[m.queryIdx].pt[0]), int(kps_l[m.queryIdx].pt[1]))
                psB = (int(kps_r[m.trainIdx].pt[0]), int(kps_r[m.trainIdx].pt[1]))
                goodMatches_pos.append([psA, psB])

        return goodMatches_pos

    def fitHomoMat(self, matches_pos):
        '''
            Fit the best homography model with MSAC algorithm - noBlending、linearBlending、linearBlendingWithConstant
        '''
        dstPoints = []  # i.e. left image(destination image)
        srcPoints = []  # i.e. right image(source image)
        for dstPoint, srcPoint in matches_pos:
            dstPoints.append(list(dstPoint))
            srcPoints.append(list(srcPoint))
        dstPoints = np.array(dstPoints)
        srcPoints = np.array(srcPoints)

        homography = Homography()

        # RANSAC parameters
        NumSample = len(matches_pos)
        threshold = 6
        NumIter = 5000
        NumRamdomSubSample = 4
        best_score = float('inf')
        Best_H = None
        best_inliers = None  # Store the inliers of the best model

        for run in range(NumIter):
            SubSampleIdx = random.sample(range(NumSample), NumRamdomSubSample)  # Random sampling
            H = homography.solve_homography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])

            # Calculate MSAC score
            proj = np.dot(H, np.hstack((srcPoints, np.ones((NumSample, 1)))).T)  # Homogeneous transformation
            proj = proj[:2, :] / proj[2, :]  # Normalize by the third (z) coordinate

            # Calculate squared distances
            dist = np.sum((proj.T - dstPoints) ** 2, axis=1)

            # Identify inliers
            inliers = dist < threshold ** 2
            score = np.sum(np.minimum(dist, threshold ** 2))

            # Update the best score, homography matrix, and inliers
            if score < best_score:
                best_score = score
                Best_H = H
                best_inliers = inliers

        # Count inliers and outliers
        num_inliers = np.sum(best_inliers)
        num_outliers = NumSample - num_inliers
        print(f"Number of inliers: {num_inliers}")
        print(f"Number of outliers: {num_outliers}")

        return Best_H

    def warp(self, imgs, HomoMat, blending_mode):
        '''
           Warp image to create panoramic image
           There are three different blending method - noBlending、linearBlending、linearBlendingWithConstant
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        stitch_img = np.zeros((max(hl, hr), wl + wr, 3),
                              dtype="int")  # create the (stitch)big image accroding the imgs height and width

        # Transform Right image(the coordination of right image) to destination iamge(the coordination of left image) with HomoMat
        inv_H = np.linalg.inv(HomoMat)
        for i in range(stitch_img.shape[0]):
            for j in range(stitch_img.shape[1]):
                coor = np.array([j, i, 1])
                img_right_coor = inv_H @ coor  # the coordination of right image
                img_right_coor /= img_right_coor[2]

                # you can try like nearest neighbors or interpolation
                y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1]))  # y for width, x for height

                # if the computed coordination not in the (hegiht, width) of right image, it's not need to be process
                if (x < 0 or x >= hr or y < 0 or y >= wr):
                    continue
                # else we need the tranform for this pixel
                stitch_img[i, j] = img_right[x, y]

        # create the Blender object to blending the image
        blender = Blender()
        if (blending_mode == "linearBlending"):
            stitch_img = blender.linearBlending([img_left, stitch_img])

        # remove the black border
        stitch_img = self.removeBlackBorder(stitch_img)

        return stitch_img

    def removeBlackBorder(self, img):
        '''
        Remove img's the black border
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1

        # bottom to top
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1

        return img[:reduced_h, :reduced_w]



class Blender:
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")

        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1

        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1

        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr))  # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j

            if (minIdx == maxIdx):  # represent this row's pixels are all zero, or only one pixel not zero
                continue

            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))

        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[
                        i, j]

        return linearBlending_img


class Homography:
    def solve_homography(self, P, m):
        try:
            A = []
            for r in range(len(P)):
                # print(m[r, 0])
                A.append([-P[r, 0], -P[r, 1], -1, 0, 0, 0, P[r, 0] * m[r, 0], P[r, 1] * m[r, 0], m[r, 0]])
                A.append([0, 0, 0, -P[r, 0], -P[r, 1], -1, P[r, 0] * m[r, 1], P[r, 1] * m[r, 1], m[r, 1]])

            u, s, vt = np.linalg.svd(A)  # Solve s ystem of linear equations Ah = 0 using SVD
            # pick H from last line of vt
            H = np.reshape(vt[8], (3, 3))
            # normalization, let H[2,2] equals to 1
            H = (1 / H.item(8)) * H
        except:
            print("Error occur!")

        return H


if __name__ == "__main__":

    fileNameList = [('1-1', '2-2')]
    for fname1, fname2 in fileNameList:
        # Read the img file
        src_path = "img/"
        fileName1 = fname1
        fileName2 = fname2
        img_left = cv2.imread(src_path + fileName1 + ".jpg")
        img_right = cv2.imread(src_path + fileName2 + ".jpg")

        # The stitch object to stitch the image
        blending_mode = "linearBlending"  # three mode - noBlending、linearBlending、linearBlendingWithConstant
        stitcher = Stitcher()
        warp_img = stitcher.stitch([img_left, img_right], blending_mode)

        # save the stitched iamge
        saveFilePath = "img/1.jpg".format(fileName1, fileName2, blending_mode)
        cv2.imwrite(saveFilePath, warp_img)