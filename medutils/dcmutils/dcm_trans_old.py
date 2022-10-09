import os
import pydicom
import numpy as np
import cv2
import time
from time import strftime, localtime
from tqdm import tqdm
from multiprocessing import Pool
from loguru import logger


class DcmTransformer:
    def __init__(self):
        self.dcm_files = []

    def load_dcm_files(self, dcm_dir, img_dir, media_type='JEG', quality=100, unit='unit8'):
        lsdir = os.listdir(dcm_dir)
        dirs = [i for i in lsdir if os.path.isdir(os.path.join(dcm_dir, i))]
        if dirs:
            for i in dirs:
                self.load_dcm_files(os.path.join(dcm_dir, i), img_dir, media_type, quality)
        files = [i for i in lsdir if os.path.isfile(os.path.join(dcm_dir, i))]
        for f in files:
            dcm_file = os.path.join(dcm_dir, f)
            if self.is_dcm_file(dcm_file):
                img_name = os.path.join(img_dir, 'CXR' + f.split('.d')[0] + '.{}'.format(media_type.lower()))
                args_tuple = (dcm_file, img_name, media_type, quality, unit)  # 添加参数tuple
                self.dcm_files.append(args_tuple)
                print('The file {} is loading...'.format(dcm_file))
        return self.dcm_files

    @staticmethod
    def is_dcm_file(dcm_file):
        file_stream = open(dcm_file, 'rb')
        file_stream.seek(128)
        data = file_stream.read(4)
        file_stream.close()
        if data == b'DICM':
            return True
        return False

    @staticmethod
    def load_dcm_meta(dcm_file):
        meta = {}
        dcm = pydicom.read_file(dcm_file)
        meta["PatientID"] = dcm.PatientID  # 患者ID
        meta["PatientName"] = dcm.PatientName  # 患者姓名
        meta["PatientBirthData"] = dcm.PatientBirthData  # 患者出生日期
        meta["PatientAge"] = dcm.PatientAge  # 患者年龄
        meta['PatientSex'] = dcm.PatientSex  # 患者性别
        meta['StudyID'] = dcm.StudyID  # 检查ID
        meta['StudyDate'] = dcm.StudyDate  # 检查日期
        meta['StudyTime'] = dcm.StudyTime  # 检查时间
        meta['InstitutionName'] = dcm.InstitutionName  # 机构名称
        meta['Manufacturer'] = dcm.Manufacturer  # 设备制造商
        meta['StudyDescription'] = dcm.StudyDescription  # 检查项目描述
        return meta

    @staticmethod
    def apply_window(img_array, win_width, win_center, rows, cols):
        img_temp = img_array
        img_temp.flags.writeable = True
        min = (2 * win_center - win_width) / 2.0 + 0.5
        max = (2 * win_center + win_width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)
        for i in np.arange(rows):
            for j in np.arange(cols):
                img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)
        min_index = img_temp < 0
        img_temp[min_index] = 0
        max_index = img_temp > 255
        img_temp[max_index] = 255
        return img_temp

    @staticmethod
    def dcm_to_img_array(dcm_file, unit='unit8'):
        ds = pydicom.dcmread(dcm_file)
        # print(ds)
        img_array = ds.pixel_array
        img_array.setflags(write=1)
        vmin = float(img_array[img_array > 0].min())  # 非零最小值
        vmax = float(img_array.max())
        newimg = np.zeros_like(img_array)
        # 对非零部分按最大值最小值归一到[0，255]
        newimg = 255 * ((img_array - vmin) / (vmax - vmin))
        if unit == 'unit8':
            newimg = newimg.astype(np.uint8)
        else:
            newimg = newimg.astype(np.uint16)
            # Now we get the good values in 16 bit format
            newimg *= 256
        invs_glag = False
        if hasattr(ds, "PresentationLUTShape"):
            if ds.PresentationLUTShape == "INVERSE":
                invs_glag = True
        if hasattr(ds, "PhotometricInterpretation"):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                invs_glag = True
        if invs_glag:
            newimg = 255 - newimg
        # newimg = np.array(newimg, dtype=np.uint16)
        # This line only change the type, not values
        # newimg *= 256  # Now we get the good values in 16 bit format
        # shape = newimg.shape
        return newimg

    def save_dcm_to_img(self, dcm_file, img_file, media_type='jpg', quality=100, unit='unit8'):
        logger.info(os.path.basename(img_file))
        img_array = self.dcm_to_img_array(dcm_file, unit)
        if media_type.lower() == 'jpg':
            cv2.imwrite(img_file, img_array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        elif media_type.lower() == 'pngs':
            quality = 9 - quality
            cv2.imwrite(img_file, img_array, [int(cv2.IMWRITE_PNG_COMPRESSION), quality])

    def trans_dcms_to_imgs(self, dcm_dir, img_dir, media_type='JEG', quality=100, unit='unit8'):
        begin_time = time.time()  # 开始时间
        # 初始化日志文件
        print('The DICOM files are loading...')
        self.load_dcm_files(dcm_dir, img_dir, media_type=media_type, quality=quality, unit=unit)
        files_num = len(self.dcm_files)
        logger.info('Total DICOM files:{}'.format(files_num))
        if files_num < 5000:
            with Pool(processes=5) as pool:
                pool.starmap(self.save_dcm_to_img, self.dcm_files)  # 多进程保存jpg文件
        else:
            prog_bar = tqdm(self.dcm_files, desc='Transforming')
            for i, (dcm_file, img_name, media_type, quality, unit) in enumerate(prog_bar):
                self.save_dcm_to_img(dcm_file, img_name, media_type, quality, unit)
                info = 'Transforming: {}/{} {}'.format(i, files_num, os.path.basename(dcm_file))
                prog_bar.set_description(info)
        total_time = time.time() - begin_time  # 结束时间
        logger.info("Total dicom files:{}, Total time : {:.6f}s".format(files_num, total_time))


dcms_dir = 'F:\\NLMCXR_samps\\dcms'
imgs_dir = 'F:\\NLMCXR_samps\\jpgs'
log_file = strftime('%Y-%m-%d-%H-%M', localtime())
log_file = os.path.join(imgs_dir, log_file + '.log')
logger.add(sink=log_file,
           level='INFO',
           format='{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}',
           enqueue=False,
           encoding='utf-8',
           # serialize=True, #序列号成JSON
           rotation="10 MB")

if __name__ == '__main__':  # 下面是将对应的dicom格式的图片转成jpg
    dcm_trsf = DcmTransformer()
    dcm_trsf.trans_dcms_to_imgs(dcms_dir, imgs_dir, media_type='jpg', quality=100, unit='unit8')
