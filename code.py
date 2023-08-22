import cv2
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class MotionDetection:
    def __init__(self, images_path: str, type_derivative_mask: str):
        self.images_path = images_path
        self.type_derivative_mask = type_derivative_mask

    def img_array(self):
        imgs_arr = []
        for img_name in os.listdir(self.images_path):
            img = cv2.imread(self.images_path + '\\' + img_name, 0)
            imgs_arr.append(np.asarray(img).astype(int))
        return imgs_arr

    def temporal_derivative(self, imgs_arr: list):
        i = 0
        temporal_derivative = []

        while i < len(imgs_arr):
            if 1 < i:
                temp_img_0 = imgs_arr[i - 2]
                temp_img_2 = imgs_arr[i]
                if self.type_derivative_mask == "prewitt":
                    temporal_derivative.append(np.subtract(temp_img_2, temp_img_0))
                elif self.type_derivative_mask == "simple":
                    temporal_derivative.append(0.5*np.subtract(temp_img_2, temp_img_0))
                else:
                    sigma = 0.3
                    x_0 = (-(-1) / (np.square(sigma))) * np.exp(-(np.square(-1)) / (2 * np.square(sigma)))
                    x_2 = (-1 / (np.square(sigma))) * np.exp(-(np.square(1)) / (2 * np.square(sigma)))
                    temporal_derivative.append(np.add(x_2 * temp_img_2, x_0 * temp_img_0))
            i += 1

        return temporal_derivative

    @staticmethod
    def filtering(filter_type: str, imgs_arr: list):
        filtered_imgs = []
        if filter_type == '3x3':
            kernel1 = np.ones((3, 3), np.float32)/9
            for i in imgs_arr:
                filtered_imgs.append(cv2.filter2D(i.astype(np.uint8), ddepth=-1, kernel=kernel1).astype(int))
        elif filter_type == '5x5':
            kernel2 = np.ones((5, 5), np.float32)/25
            for i in imgs_arr:
                filtered_imgs.append(cv2.filter2D(i.astype(np.uint8), ddepth=-1, kernel=kernel2).astype(int))

        else:
            for i in imgs_arr:
                filtered_imgs.append((cv2.GaussianBlur(i.astype(np.uint8), (5, 5), 15)).astype(int))
        return filtered_imgs

    @staticmethod
    def thresholding(temporal_derivative: list, type_of_thresh: str):

        if type_of_thresh == 'max':
            mx = 0
            mn = 0
            for td in temporal_derivative:
                if np.max(td) > mx:
                    mx = np.max(td)
                if np.min(td) < mn:
                    mn = np.min(td)
                return max(abs(mx), abs(mn))

        elif type_of_thresh == 'std':
            ideal_thresh = np.zeros(np.shape(temporal_derivative[0]))
            var_arr = np.zeros(np.shape(temporal_derivative[0]))

            for img in temporal_derivative:
                var_arr = var_arr + np.square(ideal_thresh - img)
            std_dev_arr = np.sqrt(var_arr * 1 / (len(temporal_derivative) - 1))
            average_std_dev = np.mean(std_dev_arr)
            return average_std_dev

    @staticmethod
    def combine_mask(temporal_derivative: list, imgs_arr: list, threshold: float):
        masked_imgs = []
        imgs_arr.pop(0)
        imgs_arr.pop(-1)
        i = 0
        for temp_der_i in temporal_derivative:
            temporal_derivative = np.where(abs(temp_der_i) > threshold, 1, 0)
            masked_imgs.append(np.multiply(imgs_arr[i].astype(np.uint8), temporal_derivative.astype(np.uint8)))
            i += 1
        return masked_imgs


if __name__ == "__main__":
    md = MotionDetection(r'C:\Users\udayr\PycharmProjects\CVfiles\project1\Office\Office',"gauss")
    array_of_images = md.img_array()
    filter_images = md.filtering('gauss', array_of_images)
    temp_derivative = md.temporal_derivative(filter_images)
    thresh = md.thresholding(temp_derivative[:20], "std")
    masked_images = md.combine_mask(temp_derivative, filter_images, thresh)

    for ind in range(len(masked_images)):
        cv2.imshow('images',masked_images[ind])
        cv2.waitKey(1)
        cv2.imwrite(f"Threshold output\\image{ind}.png", masked_images[ind])
