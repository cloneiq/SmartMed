import torch
import argparse
import pickle
import os
import json
from time import strftime, localtime
import torch.nn as nn
from torchvision import transforms
import datasets.nlmcxr.adv_loader as adv_loader
from datasets.dsutils.model_utils import ModelUtils
from imgrepgen.Basic2RG.models.basic_models import SentenceLSTM, WordLSTM
from tqdm import tqdm
from torch.autograd import Variable
from medutils.logutils.logger_utils import MedLogger
import modelfactory.vision.cnn_models as cnn_models
from medutils.imgutils.image_trans import ImageTrans
from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader
# from metrics.eval_report import EvalReport
from medutils.fileutils.xlsx_utils import XlsxUtils


class BasicGenRep:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.result_dir = self._init_model_dir()
        self.now_time = self._get_now()
        self.logger = self._init_logger()
        self.vocab = adv_loader.ds_vocab(mode='train')
        self.model_state_dict = self._load_mode_state_dict()
        self.cnn_model = self._init_cnn_models()
        self.sent_lstm, self.word_lstm = self._init_lstm_models()

    def _init_model_dir(self):
        result_dir = os.path.split(os.path.realpath(__file__))[0]
        result_dir = os.path.join(result_dir, self.args.save_model_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    def _init_logger(self, mode='train'):
        f_name = '{}_{}_{}_{}_{}'.format(self.now_time,
                                         self.args.cnn_model_name, self.args.mode,
                                         self.args.resize,self.args.batch_size)
        result_path = os.path.join(self.result_dir, 'report')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        log_file = os.path.join(result_path, f_name)
        return MedLogger(log_file, result_path).init_logger()

    def _load_mode_state_dict(self):
        result_dir = os.path.join(self.result_dir, 'model')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        try:
            model_path = os.path.join(result_dir, self.args.saved_model_name)
            mu = ModelUtils(model_path)
            model_state_dict = mu.load_state_dict('gpus-cpu')
            self.logger.info("[Load Model-{} Succeed!]".format(model_path))
            return model_state_dict
        except Exception as err:
            self.logger.error("[Load Model Failed] {}".format(err))
            raise err

    def _init_data_loader(self, mode='train'):
        img_trs = ImageTrans(mode=mode, resize=self.args.resize,
                             crop_size=self.args.crop_size).init_image_trans()
        data_loader = adv_loader.dsLoader(mode=mode,
                                          transform=img_trs,
                                          media_type='jpg',
                                          batch_size=self.args.batch_size,
                                          s_max=10, n_max=50)
        self.logger.info("####{} samples have been loaded!".format(len(data_loader) * self.args.batch_size))
        return data_loader

    def _init_cnn_models(self):
        # create a cnn model from factory
        cnn_model = cnn_models.create_model(self.args.cnn_model_name, pretrained=False,
                                            chan_attn=False, spat_attn=False)
        try:
            if self.model_state_dict is not None:
                self.logger.info("Visual CNN model Loaded!")
                state_dict = ModelUtils.remove_module(self.model_state_dict['cnn_model'])
                cnn_model.load_state_dict(state_dict)
        except Exception as err:
            self.logger.error("[Load Visual CNN model Failed] {}".format(err))
            raise err
        cnn_model = self._setModelToDev(cnn_model)
        return cnn_model

    def _init_lstm_models(self):
        if self.args.Parallel == 'yes':
            enc_dim = self.cnn_model.module.enc_dim
        else:
            enc_dim = self.cnn_model.enc_dim

        sent_lstm = SentenceLSTM(enc_dim, self.args.sent_hidden_dim,
                                 self.args.att_dim, self.args.sent_input_dim,
                                 self.args.word_input_dim, self.args.int_stop_dim)
        word_lstm = WordLSTM(self.args.word_input_dim, self.args.word_hidden_dim,
                             len(self.vocab), self.args.num_layers)
        try:
            if self.model_state_dict is not None:
                self.logger.info("Sentence LSTM Loaded!")
                sent_state_dict = ModelUtils.remove_module(self.model_state_dict['sent_lstm'])
                sent_lstm.load_state_dict(sent_state_dict)
                self.logger.info("Word LSTM Loaded!")
                word_state_dict = ModelUtils.remove_module(self.model_state_dict['word_lstm'])
                word_lstm.load_state_dict(word_state_dict)
        except Exception as err:
            self.logger.error("[Sentence LSTM or Word LSTM] {}".format(err))
            raise err
        sent_lstm = self._setModelToDev(sent_lstm)
        word_lstm = self._setModelToDev(word_lstm)
        return sent_lstm, word_lstm

    def _setModelToDev(self, model):
        if self.args.Parallel == 'yes':
            model = nn.DataParallel(model).to(self.device)
        else:
            model = model.to(self.device)
        return model

    def _to_var(self, x, requires_grad=True):
        x = x.to(self.device)
        return Variable(x, requires_grad=requires_grad)

    @staticmethod
    def _get_now():
        now_time = strftime('%Y-%m-%d-%H-%M', localtime())
        # now_time = time.strftime('%Y%m%d-%H-%M', time.gmtime())
        return now_time

    def gen_report(self):
        self.cnn_model.eval()
        self.sent_lstm.eval()
        self.word_lstm.eval()
        results = {}
        data_loader = self._init_data_loader(self.args.mode)
        progress_bar = tqdm(data_loader, desc='Reports are generating')
        with torch.no_grad():
            for i, (images, img_names, label, captions, prob) in enumerate(progress_bar):
                images = self._to_var(images, requires_grad=False)
                captions = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
                prob = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
                vis_enc_out = self.cnn_model(images).permute(0, 2, 3, 1)
                topics, ps = self.sent_lstm(vis_enc_out, captions, self.device)
                pred_words = torch.zeros((captions.shape[0], captions.shape[1], captions.shape[2]))
                for sent_index in range(captions.shape[1]):
                    word_outputs = self.word_lstm(topics[:, sent_index, :], captions[:, sent_index, :])
                    _, words = torch.max(word_outputs, 2)
                    pred_words[:, sent_index, :] = words
                # 循环batch_size
                for j in range(captions.shape[0]):
                    pred_caption = {}
                    target_caption = {}
                    # 循环句子
                    for k in range(captions.shape[1]):
                        if ps[j, k, 1] > 0.5:
                            words_x = pred_words[j, k, :].tolist()
                            pred = " ".join([self.vocab.idx2word[w] for w in words_x if
                                             w not in {self.vocab.word2idx['<pad>'],
                                                       self.vocab.word2idx['<start>'],
                                                       self.vocab.word2idx['<end>']}]) + "."
                            pred_caption[str(k)] = pred
                        if prob[j, k] == 1:
                            words_y = captions[j, k, :].tolist()
                            target = " ".join([self.vocab.idx2word[w] for w in words_y if
                                               w not in {self.vocab.word2idx['<pad>'],
                                                         self.vocab.word2idx['<start>'],
                                                         self.vocab.word2idx['<end>']}]) + "."

                            target_caption[str(k)] = target
                    # 取得图像编号
                    results[img_names[j]] = {
                        'Pred Sentences': pred_caption,
                        'Real Sentences': target_caption
                    }
            # print("Generate reposrt:{}".format(results))
            return self._save_predict_result(results)

    def _save_predict_result(self, result):
        f_name = '{}_{}_{}_{}_{}'.format(self.now_time, self.args.cnn_model_name,
                                         self.args.mode, self.args.resize, self.args.batch_size)
        result_path = os.path.join(self.result_dir, 'report')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        rep_json = os.path.join(result_path, f_name + '.json')
        with open(rep_json, 'w') as f:
            json.dump(result, f)
            self.logger.info('The file {} has been saved.'.format(rep_json))
        return f_name

    def gen_reports(self):
        self.args.mode = 'train'
        self.gen_report()
        self.args.mode = 'val'
        self.gen_report()
        self.args.mode = 'test'
        self.gen_report()

    # def gen_eval_metrics(self, f_name):
    #     print("Assessment metrics are generating...")
    #     result_path = os.path.join(self.result_dir, 'report')
    #     json_file = os.path.join(result_path, f_name + '.json')
    #     # evalResult = EvalReport(json_file=json_file, model_name=self.args.cnn_model_name).eval
    #     f_utils = XlsxUtils('./results/eval')
    #     rs_file = f_name + '.xls'
    #     f_utils.saveToXlsx(rs_file, 'test', evalResult)


dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Parallel', type=str, default='no')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--cnn_model_name', type=str, default='resnet50')
    parser.add_argument('--cnn_model_pretrained', type=bool, default=True)
    parser.add_argument('--int_stop_dim', type=int, default=64)
    parser.add_argument('--sent_hidden_dim', type=int, default=512)
    parser.add_argument('--sent_input_dim', type=int, default=1024)
    parser.add_argument('--word_hidden_dim', type=int, default=512)
    parser.add_argument('--word_input_dim', type=int, default=512)
    parser.add_argument('--att_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--backward_step', type=int, default=1)
    parser.add_argument('--save_model_dir', type=str, default='results')
    parser.add_argument('--saved_model_name', type=str, default='2021-03-30-07-27_resnet50_train_384_32_loss.pth.tar')

    args = parser.parse_args()
    # args.batch_size = int(args.batch_size / args.backward_step)
    genrep = BasicGenRep(dev, args)
    genrep.gen_reports()
    # genrep.gen_eval_metrics(file_name)
