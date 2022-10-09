from metrics.nlpmetrics.tokenizer.ptbtokenizer import PTBTokenizer
from metrics.nlpmetrics.bleu.bleu import Bleu
from metrics.nlpmetrics.meteor.meteor import Meteor
from metrics.nlpmetrics.rouge.rouge import Rouge
from metrics.nlpmetrics.cider.cider import Cider
from metrics.nlpmetrics.spice.spice import Spice
from metrics.nlpmetrics.wmd.wmd import WMD


class EvalCaption:
    def __init__(self, images, gts, res, scorers=None, model_name=None):
        self.evalImgs = []
        self.evalResult = {}
        self.imgToEval = {}
        self.params = {'image_id': images}
        self.gts = gts
        self.res = res
        self.scorers = scorers
        self.model_name = model_name

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # Set up scorers
        print('setting up scorers...')
        if not self.scorers:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
                (WMD(), "WMD"),
                (Spice(), "SPICE")
            ]
        # Compute scores
        self.evalResult = {'ModeName': [self.model_name]}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEvalResult(sc, m)
                    # self.eval[m] = [sc]
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEvalResult(score, method)
                # self.eval[method] = [sc]
                self.setImgToEvalImgs(scores, imgIds, method)
                print("%s: %0.3f" % (method, score))
            print(self.evalResult)
        self.setEvalImgs()

    def setEvalResult(self, score, method):
        self.evalResult[method] = [score]

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            # if not imgId in self.imgToEval:
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, evl in self.imgToEval.items()]


if __name__ == '__main__':
    rng = range(2)
    # 数据样式（预测字典，真实值字典）
    datasetGTS = {
        0: [{'image_id': 0, 'caption': 'the man is playing a guitar'}],
        1: [{'image_id': 1, 'caption': 'a woman is cutting vegetables'}]
    }
    datasetRES = {
        0: [{'image_id': 0, 'caption': 'man is playing guitar'}],
        1: [{'image_id': 1, 'caption': 'a woman is cutting vegetables'}]
    }
    print("gts :{}".format(datasetGTS))
    print("res :{}".format(datasetRES))
    scorer = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (WMD(), "WMD")
    ]
    evalObj = EvalCaption(rng, datasetGTS, datasetRES, scorer)
    # evalObj = EvalCaption(rng, datasetGTS, datasetRES)
    evalObj.evaluate()
    print(evalObj.eval)
