from metrics.nlpmetrics.wmd.wmd import WMD
gts = {0: ['the man is sitting on the graph'], 1: ['the man is sitting on the graph']}
res = {0: ['the man is sitting on the graph'], 1: ['the man is sitting on the graph']}
scorer = WMD()
scores = scorer.calc_score(gts,res)
print(scorer)


