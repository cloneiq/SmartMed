import medutils.dcmutils.dcm_trans as dcm_trans
from time import strftime, localtime
import os
from loguru import logger
dcms_dir = 'F:\\NLMCXR_samps\\dcms'
imgs_dir = 'F:\\NLMCXR_samps\\jpgs'
# log_file = strftime('%Y-%m-%d-%H-%M', localtime())
# log_file = os.path.join(imgs_dir, log_file + '.log')
# logger.add(sink=log_file,
#            level='INFO',
#            format='{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}',
#            enqueue=False,
#            encoding='utf-8',
#            # serialize=True, #序列号成JSON
#            rotation="10 MB")
media_type = 'JEG'
quality = 100
unit = 'unit8'

dcm_trans.trans_dcms_to_imgs(dcms_dir,imgs_dir,media_type,quality,unit)