import os
import pydicom
import numpy as np
import cv2
import time
from time import strftime, localtime
from tqdm import tqdm
from multiprocessing import Pool
from loguru import logger


# 判断一个文件是否是DICOM文件
def is_dcm_file(dcm_file):
    file_stream = open(dcm_file, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False


def load_dcm_files(dcm_dir='dcms'):
    dcm_files = []
    for root, dirs, files in os.walk(dcm_dir):
        # for dir1 in dirs:
        #  print(os.path.join(root, dir1))
        for file in files:
            dcm_file = os.path.join(root, file)
            if is_dcm_file(dcm_file):
                args_tuple = (dcm_file, None)
                dcm_files.append(args_tuple)
                print('The file {} is loading...'.format(dcm_file))
    return dcm_files


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


def apply_LUT(self, img, hdr):
    """
            Apply LUT specified in header to the image, if the header specifies one.
            Specification:
            http://dicom.nema.org/medical/dicom/2017a/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.1
            """
    lut_seq = getattr(hdr, "VOILUTSequence", None)
    if lut_seq is None:
        # print("No LUT for image {}".format(generate_unique_filename(hdr)))
        return img, hdr
    # Use the first available LUT:
    lut_desc = getattr(lut_seq[0], "LUTDescriptor", None)
    lut_data = getattr(lut_seq[0], "LUTData", None)
    if lut_desc is None or lut_data is None:
        return img, hdr
    try:
        first_value = int(lut_desc[1])
    except:
        pass
    bit_depth = int(lut_desc[2])
    sign_selector = "u" if type(first_value) == int and first_value >= 0 else ""
    type_selector = 8
    while type_selector < bit_depth and type_selector < 64:
        type_selector *= 2
    orig_type = img.dtype

    img = np.round(img)

    if type(first_value) != int:
        first_value = img.min()

    LUT = {
        int(v): lut_data[j]
        for j, v in [(i, first_value + i) for i in range(len(lut_data))]
    }

    img2 = np.array(img)
    img2 = img2.astype("{}int{}".format(sign_selector, type_selector))
    img2[img < first_value] = first_value
    img2 = np.vectorize(lambda x: LUT[int(x)])(img2)
    img2[img >= (first_value + len(lut_data))] = lut_data[-1]

    del hdr.VOILUTSequence

    return img2.astype(orig_type), hdr


def apply_window(img, hdr):
    """
            Apply intensity window as defined in the DICOM header to the image (if any window
            is defined).
            This is applied after any LUT and rescale/intercept.
            See https://www.dabsoft.ch/dicom/3/C.11.2.1.2/
            This implementation will set the output range (min, max) equal to the
            input range (original min, max).  If scaling is desired, do that after calling
            this function.
            """
    window_center = getattr(hdr, "WindowCenter", None)
    if window_center is None:
        return img, hdr
    y_min = img.min()
    y_max = img.max()
    window_width = getattr(hdr, "WindowWidth", None)
    window_center, window_width = float(window_center), float(window_width)

    img_out = np.zeros_like(img)

    # y = ((x - (c - 0.5)) / (w-1) + 0.5) * (y max - y min )+ y min
    img_out = ((img - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
            y_max - y_min
    ) + y_min
    #  if (x <= c - 0.5 - (w-1)/2), then y = y min
    img_out[img <= (window_center - 0.5 - (window_width - 1) / 2.0)] = y_min
    # else if (x > c - 0.5 + (w-1)/2), then y = y max ,
    img_out[img > (window_center - 0.5 + (window_width - 1) / 2.0)] = y_max

    return img_out, hdr


def read_dicom_raw(self, dcm_file):
    dcm = pydicom.read_file(dcm_file)
    img = dcm.pixel_array
    return img, dcm


def read_dicom(self, dcm_file):
    img, dcm = self.read_dicom_raw(dcm_file)
    img, dcm = self.rescale_image(img, dcm)
    img, dcm = self.apply_LUT(img, dcm)
    img, dcm = self.apply_window(img, dcm)
    img = img.astype(float)
    img -= img.min()
    img /= img.max()
    img *= 255.0
    # Convert to 8-bit unsigned int:
    img = img.astype("uint8")
    return img, dcm


def needs_rescale(self, hdr):
    return hasattr(hdr, "RescaleSlope") or hasattr(hdr, "RescaleIntercept")


def rescale_image(self, img, hdr):
    """Apply rescale formula from DICOM header, if that information is available."""
    if not self.needs_rescale(hdr):
        return (img, hdr)
    if type(hdr) == type([]):
        hdr = hdr[0]
    img = np.array(img)
    img_type = img.dtype
    # Get the scaling info
    rescale_slope = float(getattr(hdr, "RescaleSlope", 1))
    rescale_intercept = float(getattr(hdr, "RescaleIntercept", 0))
    # Re-Scale
    img = img.astype(np.float64) * rescale_slope + rescale_intercept
    img = img.astype(img_type)
    # Update the header
    setattr(hdr, "RescaleSlope", 1.0)
    setattr(hdr, "RescaleIntercept", 0.0)
    return (img, hdr)

    # 调窗


def apply_window_array(img_array, win_width, win_center, rows, cols):
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


def apply_window_file(self, dcm_file, win_width, win_center, rows, cols):
    img_array = self.dcm_to_img_array(dcm_file)
    apped_img = self.apply_window_array(img_array, win_width, win_center, rows, cols)
    return apped_img


def dcm_to_img_array(dcm_file, unit='unit8'):
    ds = pydicom.dcmread(dcm_file)
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
    if is_need_invert(ds):
        newimg = 255 - newimg
    # newimg = np.array(newimg, dtype=np.uint16)
    # This line only change the type, not values
    # newimg *= 256  # Now we get the good values in 16 bit format
    # shape = newimg.shape
    return newimg


def is_need_invert(dcm):
    invt_flag = False
    if hasattr(dcm, "PresentationLUTShape"):
        if dcm.PresentationLUTShape == "INVERSE":
            invt_flag = True
    if hasattr(dcm, "PhotometricInterpretation"):
        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            invt_flag = True
    return invt_flag


class DcmTransformer:
    def __init__(self, media_type='JPG', quality=100, unit='unit8',img_size=None):
        self.dcm_dir = 'dcms'
        self.img_dir = 'jpgs'
        self.media_type = media_type
        self.quality = quality
        self.unit = unit
        self.img_size = img_size

    # 将一个DICOM文件转成指定图像
    def save_dcm_to_img(self, dcm_file, img_file=None):
        if img_file is None:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
            tmp_file = os.path.basename(dcm_file)
            img_file = os.path.join(self.img_dir,
                                    'CXR' + tmp_file.split('.d')[0] + '.{}'.format(self.media_type.lower()))
        logger.info(os.path.basename(img_file))
        img_array = dcm_to_img_array(dcm_file, self.unit)
        if self.img_size is not None:
            h, w = img_array.shape[0:2]
            # 等比例放缩
            if h < w:
                new_w = self.img_size
                new_h = (new_w*h)//w
            else:
                new_h = self.img_size
                new_w = (new_h*w)//h
            # INTER_NEAREST
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if self.media_type.lower() == 'jpg':
            cv2.imwrite(img_file, img_array, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        elif self.media_type.lower() == 'pngs':
            cv2.imwrite(img_file, img_array, [int(cv2.IMWRITE_PNG_COMPRESSION), 9 - self.quality])

    def trans_dcms_to_imgs(self, dcm_dir, img_dir):
        self.dcm_dir = dcm_dir
        self.img_dir = img_dir
        begin_time = time.time()  # 开始时间
        # 初始化日志文件
        print('The DICOM files are loading...')
        dcm_files = load_dcm_files(self.dcm_dir)
        files_num = len(dcm_files)
        logger.info('Total DICOM files:{}'.format(files_num))
        if files_num > 5000:
            with Pool(processes=5) as pool:
                pool.starmap(self.save_dcm_to_img, dcm_files)  # 多进程保存jpg文件
        else:
            prog_bar = tqdm(dcm_files, desc='Transforming')
            for i, (dcm_file, img_name) in enumerate(prog_bar):
                self.save_dcm_to_img(dcm_file)
                info = 'Transforming: {}/{} {}'.format(i, files_num, os.path.basename(dcm_file))
                prog_bar.set_description(info)
        total_time = time.time() - begin_time  # 结束时间
        logger.info("Total dicom files:{}, Total time : {:.6f}s".format(files_num, total_time))


dcms_dir = 'F:\\NLMCXR_DCMs'
imgs_dir = 'F:\\NLMCXR_JPGs'
# log_dir = os.path.split(os.path.realpath(__file__))[0]
# log_dir = os.path.join(log_dir, 'logs')
log_file = strftime('%Y-%m-%d-%H-%M', localtime())
log_file = os.path.join(imgs_dir, log_file + '.log')
logger.add(sink=log_file,
           level='INFO',
           format='{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}',
           enqueue=False,
           encoding='utf-8',
           # serialize=True, #序列号成JSON
           rotation="10 MB")


def trans_dcms_to_imgs(dcm_dir, img_dir, media_type='jpg', quality=100, unit='unit8'):
    dcm_trfm = DcmTransformer(media_type=media_type, quality=quality, unit=unit)
    dcm_trfm.trans_dcms_to_imgs(dcm_dir, img_dir)


if __name__ == '__main__':
    dcm_trfm = DcmTransformer(media_type='jpg', quality=100, unit='unit8', img_size=2048)
    dcm_trfm.trans_dcms_to_imgs(dcms_dir, imgs_dir)
