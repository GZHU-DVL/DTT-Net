'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
from imageio import imread

class metric:
    def __init__(self, img_dir):
        """Initialize the metric classes

        Parameters:
            img_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.img_dir = img_dir


    def main(self):
        # Configurations

        # ret_Or = self.getListFiles(os.path.join(self.img_dir, 'Or'))
        ret_pred = self.getListFiles(os.path.join(self.img_dir, 'pred_Br'))
        ret = self.getListFiles(os.path.join(self.img_dir, 'Br'))
        listSpicalpath = set()
        new_ret_pred = []
        new_ret = []
        new_ret_Or = []
        suffix = ['.png']
        # for index in ret_Or:
        #     new_ret_Or += index.split('_Or.png', 1)[:1]  # 去掉_alpha后面的部分
        for index in ret_pred:
            new_ret_pred += index.split('_pred_Br.png', 1)[:1]  # 去掉_alpha后面的部分
        for index in ret:
            new_ret += index.split('_Br.png', 1)[:1]  # 去掉_alpha后面的部分
        new_ret_Or = ['{}{}'.format(a, b) for b in suffix for a in new_ret_Or]  # 回补后缀
        new_ret_pred = ['{}{}'.format(a, b) for b in suffix for a in new_ret_pred]  # 回补后缀
        new_ret = ['{}{}'.format(a, b) for b in suffix for a in new_ret]  # 回补后缀
        for i in range(len(ret)):
            # os.rename(ret_Or[i], new_ret_Or[i])
            os.rename(ret[i], new_ret[i])
            os.rename(ret_pred[i], new_ret_pred[i])

        # GT - Ground-truth;
        # Gen: Generated / Restored / Recovered images
        folder_GT = os.path.join(self.img_dir,"Br")
        folder_Gen = os.path.join(self.img_dir,"pred_Br")
        crop_border = 4  # same with scale
        suffix = ''  # suffix for Gen images
        test_Y = False  # True: test Y channel only; False: test RGB channels

        PSNR_all = []
        SSIM_all = []
        img_list = sorted(glob.glob(folder_GT + '/*'))

        if test_Y:
            print('Testing Y channel.')
        else:
            print('Testing RGB channels.')

        for i, img_path in enumerate(img_list):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            im_GT = imread(img_path) / 255.
            im_Gen = imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.

            if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                im_GT_in = self.bgr2ycbcr(im_GT)
                im_Gen_in = self.bgr2ycbcr(im_Gen)
            else:
                im_GT_in = im_GT
                im_Gen_in = im_Gen

            # crop borders
            if crop_border == 0:
                cropped_GT = im_GT_in
                cropped_Gen = im_Gen_in
            else:
                if im_GT_in.ndim == 3:
                    cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                    cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
                elif im_GT_in.ndim == 2:
                    cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
                else:
                    raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

            # calculate PSNR and SSIM
            # PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
            PSNR = self.calculate_rgb_psnr(cropped_GT * 255, cropped_Gen * 255)

            SSIM = self.calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
            print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
                i + 1, base_name, PSNR, SSIM))
            PSNR_all.append(PSNR)
            SSIM_all.append(SSIM)

        print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
            sum(PSNR_all) / len(PSNR_all),
            sum(SSIM_all) / len(SSIM_all)))

    def getListFiles(self, path):
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))
        return ret

    def mkdir(path):
        """create a single empty directory if it didn't exist

        Parameters:
            path (str) -- a single directory path
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def calculate_psnr(self,img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calculate_rgb_psnr(self,img1, img2):
        """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
        """
        n_channels = np.ndim(img1)
        sum_psnr = 0
        for i in range(n_channels):
            this_psnr = self.calculate_psnr(img1[:, :, i], img2[:, :, i])
            sum_psnr += this_psnr
        return sum_psnr / n_channels

    def ssim(self,img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_ssim(self,img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(img1.shape[2]):
                    ssims.append(self.ssim(img1[..., i], img2[..., i]))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def bgr2ycbcr(img, only_y=True):
        '''same as matlab rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                                  [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)



# choose





