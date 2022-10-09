from metrics.eval_report import EvalReport
from metrics.nlpmetrics.tokenizer.ptbtokenizer import PTBTokenizer
from metrics.nlpmetrics.bleu.bleu import Bleu
from metrics.nlpmetrics.meteor.meteor import Meteor
from metrics.nlpmetrics.rouge.rouge import Rouge
from metrics.nlpmetrics.cider.cider import Cider
from metrics.nlpmetrics.spice.spice import Spice
from metrics.nlpmetrics.wmd.wmd import WMD
from medutils.fileutils.xlsx_utils import XlsxUtils
import os


class BasicEval:
    @staticmethod
    def eval(json_file):
        file_name = json_file.split('.')[0]
        file_comp = file_name.split('_')
        mod_name = file_comp[1]
        mode = file_comp[2]
        json_file = os.path.join('./results/report', json_file)
        print('The scorers are loading...')
        scorer = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
            (WMD(), "WMD")
        ]
        result = EvalReport(json_file=json_file, scorers=scorer, model_name=mod_name).eval()
        file_utils = XlsxUtils('./results/eval')
        file_utils.saveToXlsx(file_name + '.xls', mode, result)


if __name__ == '__main__':

    eval_file = '2021-04-29-06-16_resnet50_train_384_32.json'
    BasicEval.eval(eval_file)
    eval_file = '2021-04-29-06-16_resnet50_val_384_32.json'
    BasicEval.eval(eval_file)
    eval_file = '2021-04-29-06-16_resnet50_test_384_32.json'
    BasicEval.eval(eval_file)
    # eval_file = '2021-02-09_06_40_resnet152_val.json'
    # model_name = 'resnet152'
    # scorer = [
    #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    #     (Meteor(), "METEOR"),
    #     (Rouge(), "ROUGE_L"),
    #     (Cider(), "CIDEr"),
    #     # (Spice(), "SPICE"),
    #     (WMD(), "WMD")
    # ]
    # # evalObj = EvalReport(eval_file, scorer)
    # # json_file = None, val_data = None, scorers = None, model_name = None
    # evalResult = EvalReport(json_file=eval_file, scorers=scorer, model_name=model_name).eval()
    # # evalResult = {'ModeName': ['张三'],
    # #         'BLEU-1': [80],
    # #         'BLEU-2': [80],
    # #         'BLEU-3': [80],
    # #         'BLEU-4': [80],
    # #         'METEOR': [0.354],
    # #         'ROUGE_L': [0.6],
    # #         'CIDEr': [0.3],
    # #         'WMD': [0.2],
    # #         'SPICE': [0.6]
    # #         }
    # f_utils = XlsxUtils('./results/eval')
    # f_utils.saveToXlsx('2021-02-09_06_40_resnet152_val.xls', 'val', evalResult)
    # # print(evalResult)
