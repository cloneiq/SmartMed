# coding:utf-8
'''
TODO: 将png图像转为dicom
author:chenhao
email:haochen9212@outlook.com
data:2018.10.16
'''
import cv2
import pydicom as dicom
import os
import numpy as np


def _png2dicom(imgpng, dcm):
    dcm.PixelData = imgpng
    return dcm


def _dcm2png(dcm):
    img = dcm.pixel_array
    return img


def getfile(fileimg, filedcm):
    img = cv2.imread(fileimg, cv2.IMREAD_UNCHANGED)
    dcm = dicom.read_file(filedcm)
    return img, dcm


def getfilelist(path):
    dcmlist = list(map(lambda x: path + 'dcm/' + x, os.listdir(path + 'dcm')))
    pnglist = list(map(lambda x: path + 'pngs/' + x, os.listdir(path + 'pngs')))
    length = len(dcmlist)
    return pnglist, dcmlist, length


def png2dicom(path, savepathdir):
    imglist, dcmlist, length = getfilelist(path)
    for i in range(length):
        print('{} is processing'.format(dcmlist[i].split('/')[-1]))
        img, dcm = getfile(imglist[i], dcmlist[i])
        new_dcm = _png2dicom(img, dcm)
        savepath = os.path.join(savepathdir, dcmlist[i].split('/')[-1])
        new_dcm.save_as(savepath)
        print('{} is processed'.format(dcmlist[i].split('/')[-1]))


def dcm2png(path, savepathdir):
    lists = os.listdir(path)
    for file in lists:
        print('{} is processing'.format(file))
        dcm_path = os.path.join(path, file)
        dcm = dicom.read_file(dcm_path)
        img = _dcm2png(dcm)
        img = img.astype(np.uint16)
        name = file.split('.')[0] + '.pngs'
        img_path = os.path.join(savepathdir, name)
        # numpngw.write_png(img_path, img)
        print('{} is processed'.format(file))


'''
TODO: 判断是否需要对图像反色处理
author: chenhao
data:2018.11.13
'''

import numpy as np
import cv2
import os


def getCH(img, nbins=256):
    '''
    TODO: get img Cumulative distribution histogram
    :param img: ndarray input image
    :param nbins: integer histogram bins
    :return ch： ndarray result of Cumulative distribution histogram
    '''
    # get image histogram
    imgmax = img.max()
    imgmin = img.min()

    hist, bins = np.histogram(img.flatten(), nbins, [imgmin, imgmax])

    area = img.shape[0] * img.shape[1]
    # calculate cumulative histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf / area  # get normalized cumulative distribution histogram

    return cdf_normalized, bins


def getMajorGrey(nbins, cd_normalized, th):
    '''
    TODO: get img majority object grey intensity distribution
    :param img:
    :param nbins:
    :param cd_normalized:
    :return:
    '''

    th_min = th
    th_max = 1.0 - th

    cd_normalized = np.array(cd_normalized)
    index_th_min = np.where(cd_normalized > th_min)[0][0]
    index_th_max = np.where(cd_normalized > th_max)[0][0]

    major_min = nbins[index_th_min].astype(np.uint16)
    major_max = nbins[index_th_max].astype(np.uint16)

    return major_max, major_min


def _judge_inverse(img, max):
    '''
    TODO: 根据四角判断是否局域主要成分的上位，如果是就进行反色，返回flag
    :param img:
    :param max:
    :param min:
    :return:
    '''
    row, col = img.shape
    block1 = img[0:10, 0:10]
    block2 = img[row - 10:row, 0:10]
    block3 = img[0:10, col - 10:col]
    block4 = img[row - 10:row, col - 10:col]

    m1 = np.mean(block1)
    m2 = np.mean(block2)
    m3 = np.mean(block3)
    m4 = np.mean(block4)

    Flag = False
    if m1 >= max:
        Flag = True
    if m2 >= max:
        Flag = True
    if m3 >= max:
        Flag = True
    if m4 >= max:
        Flag = True
    return Flag


def judge_inverse(img):
    cdf, bins = getCH(img, 256)
    grey_max, grey_min = getMajorGrey(bins, cdf, 0.05)
    flag = _judge_inverse(img, grey_max)
    return flag


def dicomtrans_one(ds):
    '''
    Transfer one dicom image
    '''
    img = ds.pixel_array
    img.setflags(write=1)
    # 如果存在线形转换标签，则进行线形转换
    # if hasattr(ds,"RescaleIntercept") and hasattr(ds,"RescaleSlope"):
    #     img = ds.RescaleSlope * img + ds.RescaleIntercept
    # if hasattr(ds,"WindowCenter") and hasattr(ds,"WindowWidth"):
    #     vcenter = ds.WindowCenter
    #     vwidth = ds.WindowWidth
    #     if isinstance(vcenter, dicom.multival.MultiValue) and isinstance(vwidth, dicom.multival.MultiValue):
    #         vwidth = int(vwidth[0])
    #         vcenter = int(vcenter[0])
    #     vmin = vcenter - vwidth / 2
    #     vmax = vcenter + vwidth / 2
    #     img[img > vmax] = vmax
    #     img[img < vmin] = vmin
    #     # if ds[0x0028, 0x0100].value == 8:
    #     img = 255.0 * (img - vmin) / (vmax - vmin)
    #     img = img.astype(np.uint8)
    #     # else:
    #     #     img = 65536.0 * (img - vmin) / (vmax - vmin)
    #     #     img = img.astype(np.uint16)
    # else:
    # 根据PresentationLUTShape 判断是否需要取反
    Flag = False
    if hasattr(ds, "PresentationLUTShape"):
        if ds.PresentationLUTShape == "INVERSE":
            Flag = True
    # Flag = judge_inverse(ds.pixel_array)
    vmin = float(img[img > 0].min())  # 非零最小值
    vmax = float(img.max())
    newimg = np.zeros_like(img)
    # 对非零部分按最大值最小值归一到【0，255】
    newimg = 255.0 * (img - vmin) / (vmax - vmin)
    if Flag:
        # 取反
        newimg = 255.0 - newimg
        # 将img为0的部分 设置为vmax
        newimg[img == 0] = 255.0
        # 转成整形
        newimg = newimg.astype(np.uint8)
        img = newimg

    return img


def main():
    dicompath = 'dcms/2/'
    outpath = 'pngs/'
    refer_path = 'pngs/'
    dcm = dicom.read_file(os.path.join(dicompath, '2_IM-0652-1001.dcm'))
    img = dicomtrans_one(dcm)
    cv2.imwrite('bbb.pngs', img)



if __name__ == "__main__":
    main()