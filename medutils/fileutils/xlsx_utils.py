import os
import pandas as pd


class XlsxUtils:
    def __init__(self, file_dir):
        self.file_dir = file_dir

    def saveToXlsx(self, file_name, sheet_name, xlsx_data):
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
        file_name = os.path.join(self.file_dir, file_name)
        df = pd.DataFrame(xlsx_data)
        df.to_excel(file_name, sheet_name=sheet_name, index=False)


if __name__ == '__main__':
    f_utils = XlsxUtils('test')
    dic1 = {'ModeName': ['张三'],
            'BLEU-1': [80],
            'BLEU-2': [80],
            'BLEU-3': [80],
            'BLEU-4': [80],
            'METEOR': [0.354],
            'ROUGE_L': [0.6],
            'CIDEr': [0.3],
            'WMD': [0.2],
            'SPICE': [0.6]
            }
    f_utils.saveToXlsx('2.xls', 'test', dic1)
