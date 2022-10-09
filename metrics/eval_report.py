import json
from metrics.eval_caption import EvalCaption
from metrics.nlpmetrics.bleu.bleu import Bleu
from metrics.nlpmetrics.cider.cider import Cider
from metrics.nlpmetrics.meteor.meteor import Meteor
from metrics.nlpmetrics.rouge.rouge import Rouge
from metrics.nlpmetrics.spice.spice import Spice
from metrics.nlpmetrics.wmd.wmd import WMD


# 评价生成报告的几个指标
class EvalReport:
    def __init__(self, json_file=None, val_data=None, scorers=None, model_name=None):
        self.eval_file = json_file
        self.val_data = val_data
        self.scorers = scorers
        self.model_name = model_name

    def eval(self):
        rng, gts, res = self._init_eval_data()
        if self.model_name is None:
            self.model_name = 'unknown'
        evalcap = EvalCaption(rng, gts, res, self.scorers, model_name=self.model_name)
        evalcap.evaluate()
        return evalcap.evalResult

    def _init_eval_data(self):
        print("Loading the eval file {}...".format(self.eval_file))
        if self.eval_file:
            with open(self.eval_file, 'r') as f:
                data = json.load(f)
        elif self.val_data:
            data = self.val_data
        else:
            print("error")
        # 预测出来的句子
        datasetGTS = {}
        # 真实预测句子
        datasetRES = {}
        print("Parsing the data from eval file {}...".format(self.eval_file))
        for i, image_id in enumerate(data):
            arrayGTS = []
            for each in data[image_id]['Real Sentences']:
                sent = data[image_id]['Real Sentences'][each]
                if sent.strip():
                    arrayGTS.append(sent)
                real_sent = '. '.join(arrayGTS)
            datasetGTS[i] = [{'image_id': i, 'caption': real_sent}]
            arrayRES = []
            for each in data[image_id]['Pred Sentences']:
                sent = data[image_id]['Pred Sentences'][each]
                if sent.strip():
                    arrayRES.append(sent)
                pred_sent = '. '.join(arrayRES)
            datasetRES[i] = [{'image_id': i, 'caption': pred_sent}]

        rng = range(len(data))
        # print("gts :{}".format(datasetGTS))
        # print("res :{}".format(datasetRES))
        return rng, datasetGTS, datasetRES


if __name__ == '__main__':
    eval_file = 'eval.json'
    # scorer = [
    #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    #     # (Meteor(), "METEOR"),
    #     # (Rouge(), "ROUGE_L"),
    #     # (Cider(), "CIDEr"),
    #     # (Spice(), "SPICE"),
    #     (WMD(), "WMD")
    # ]
    # evalObj = EvalReport(eval_file, scorer)
    evalObj = EvalReport(json_file=eval_file)
    print(evalObj.eval())
