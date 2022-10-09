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


class DGLNetEval:
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
    eval_file = '2022-07-01-01-11_resnet50_train_384_24.json'
    DGLNetEval.eval(eval_file)
    eval_file = '2022-07-01-01-11_resnet50_val_384_24.json'
    DGLNetEval.eval(eval_file)
    eval_file = '2022-07-01-01-11_resnet50_test_384_24.json'
    DGLNetEval.eval(eval_file)

