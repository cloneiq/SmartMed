import torch
import argparse
import pickle
import os
import json
from time import strftime, localtime
import torch.nn as nn
from torchvision import transforms
import datasets.nlmcxr.adv_big_loader as adv_big_loader
from datasets.dsutils.model_utils import ModelUtils
from imgrepgen.DGLNet.models.cogl_net import CoGLNet
from imgrepgen.DGLNet.models.lang_net import SentenceLSTM, WordLSTM
from tqdm import tqdm
from torch.autograd import Variable
from medutils.logutils.logger_utils import MedLogger
import modelfactory.vision.cnn_models as cnn_models
from medutils.imgutils.image_trans import ImageTrans
from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader
from imgrepgen.DGLNet.models.coglg_utils import DGLNetUtils

class DGLNetGenRep:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.result_dir = self._init_model_dir()
        self.now_time = self._get_now()
        self.logger = self._init_logger()
        self.vocab = adv_big_loader.ds_vocab(mode='train')
        self.model_state_dict = self._load_mode_state_dict()
        self.vis_model = self._init_vis_models()
        self.sent_lstm, self.word_lstm = self._init_lstm_models()
        self.load_batch_size = int(args.batch_size / args.backward_step)

    def _init_model_dir(self):
        result_dir = os.path.split(os.path.realpath(__file__))[0]
        result_dir = os.path.join(result_dir, self.args.save_model_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    def _init_logger(self, mode='train'):
        f_name = '{}_{}_{}_{}_{}'.format(self.now_time,
                                         self.args.model_name, self.args.mode,
                                         self.args.resize, self.args.batch_size)
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

    # def _init_data_loader(self, mode='train'):
    #     data_loader = adv_big_loader.dsLoader(mode=mode,
    #                                           # transform=img_trs,
    #                                           batch_size=self.load_batch_size,
    #                                           s_max=10, n_max=50)
    #     self.logger.info("####{} samples have been loaded!".format(len(data_loader) * self.args.batch_size))
    #     return data_loader

    def _init_data_loader(self, mode='train'):
        data_loader = adv_big_loader.dsLoader(mode=mode,
                                              media_type=self.args.media_type,
                                              batch_size=self.load_batch_size,
                                              s_max=10, n_max=50)
        self.logger.info("####{} samples have been loaded!".format(len(data_loader) * self.args.batch_size))
        return data_loader

    def _init_vis_models(self):
        vis_model = CoGLNet(self.args.model_name, self.args.chan_attn, self.args.spat_attn, self.args.mode_name)
        try:
            if self.model_state_dict is not None:
                self.logger.info("Visual CNN model Loaded!")
                state_dict = ModelUtils.remove_module(self.model_state_dict['vis_model'])
                vis_model.load_state_dict(state_dict)
        except Exception as err:
            self.logger.error("[Load Visual CNN model Failed] {}".format(err))
            raise err
        vis_model = self._set_model_to_dev(vis_model)
        return vis_model

    def _init_lstm_models(self):
        if self.args.Parallel == 'yes':
            enc_dim = self.vis_model.module.cogl_net.enc_dim
        else:
            enc_dim = self.vis_model.cogl_net.enc_dim

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
        sent_lstm = self._set_model_to_dev(sent_lstm)
        word_lstm = self._set_model_to_dev(word_lstm)
        return sent_lstm, word_lstm

    def _set_model_to_dev(self, model):
        if self.args.Parallel == 'yes':
            model = nn.DataParallel(model).to(self.device)
        else:
            model = model.to(self.device)
        return model

    def _to_var(self, x, requires_grad=True):
        x = x.to(self.device)
        return Variable(x, requires_grad=requires_grad)

    def __init_transform(self, mode='train'):
        tsf = ImageTrans(mode='train', resize=self.args.resize,
                         crop_size=self.args.crop_size).init_image_trans()
        return tsf

    @staticmethod
    def _get_now():
        now_time = strftime('%Y-%m-%d-%H-%M', localtime())
        return now_time

    # 转换成tensor
    @staticmethod
    def imgs_trans(imgs, transform):
        # PIL images conver to tensor
        out_imgs = []
        for img in imgs:
            out_imgs.append(transform(img))
        out_imgs = torch.stack(out_imgs, dim=0)
        return out_imgs

        #

    # 将patches转换为tensor
    # 将patches转换为tensor
    def patches_trans(self, patches, max_patches, csys, transform):
        loc_imgs = torch.zeros(self.load_batch_size, max_patches, 3, self.args.patch_size, self.args.patch_size)
        for j in range(len(patches)):
            ts_patches = self.imgs_trans(patches[j], transform)
            patch_num = ts_patches.size(0)
            loc_imgs[j][:patch_num] = ts_patches
            csys[j] += [[-1, -1]] * (max_patches - len(csys[j]))
        return loc_imgs, csys

    def imgs_to_patches(self, src_img, transform):
        patch_size = (self.args.patch_size, self.args.patch_size)
        patches, sizes, ratios, max_patches = utils.imgs_to_patches(src_img, patch_size)
        # print("Max Patches: ", max_patches)
        # loc_imgs = torch.zeros(self.load_batch_size, max_patches, 3, self.args.patch_size, self.args.patch_size)
        loc_imgs = torch.zeros(self.load_batch_size, max_patches, 3, self.args.patch_size, self.args.patch_size)
        for j in range(len(src_img)):
            # patches[j] = self.resize(patches[j], (self.args.patch_size, self.args.patch_size))
            ts_patches = self.imgs_trans(patches[j], transform)
            patch_num = ts_patches.size(0)
            loc_imgs[j][:patch_num] = ts_patches
        return loc_imgs

    # 调整图像大小
    @staticmethod
    def img_resize(images, shape=None):
        resized = list(images)
        for i in range(len(images)):
            w, h = images[i].size
            if shape is None:
                if w > h:
                    img_size = (h, h)
                else:
                    img_size = (w, w)
            else:
                img_size = shape
            # print('image size:----{}'.format(img_size))
            resized[i] = images[i].resize(img_size)
        return resized

    # 通过一个原始图像生成一个报告
    def gen_one_report(self, org_imgs):
        self.vis_model.eval()
        self.sent_lstm.eval()
        self.word_lstm.eval()
        results = {}
        transform = self.__init_transform('train')
        # org_imgs转换为图像列表
        with torch.no_grad():
            if self.args.mode_name == 'glb':
                ts_imgs = self.imgs_trans(org_imgs, transform)
                ts_imgs = self._to_var(ts_imgs)
                vis_enc_out = self.vis_model.forward(ts_imgs)
            vis_enc_out = vis_enc_out.permute(0, 2, 3, 1)

    def gen_reports(self, mode):
        self.vis_model.eval()
        self.sent_lstm.eval()
        self.word_lstm.eval()
        self.mode = mode
        transform = self.__init_transform('train')
        results = {}
        data_loader = self._init_data_loader(mode)
        progress_bar = tqdm(data_loader, desc='Reports are generating')
        with torch.no_grad():
            for i, (org_imgs, img_names, label, captions, prob) in enumerate(progress_bar):
                org_imgs = self.img_resize(org_imgs)
                print(img_names)
                # org_imgs：是原始图像，不是tensor，在进行多GPU并行时，无法进进行数据分割，而是在每个GPU上复制一份
                # 视觉特征抽取、选择与融合
                vis_enc_out = None
                batch_size = None
                # glb:全局网络模式
                if self.args.mode_name.lower() == 'glb':
                    ts_imgs = self.imgs_trans(org_imgs, transform)
                    batch_size, c, w, h = ts_imgs.size()
                    ts_imgs = ts_imgs.view(batch_size, -1, c, w, h)
                    ts_imgs = self._to_var(ts_imgs, requires_grad=False)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs)
                # loc:局部网络模式
                elif self.args.mode_name.lower() == 'loc':
                    # 将一个原始图像分割成块(imgs_in->patches)
                    patch_size = (self.args.patch_size, self.args.patch_size)
                    patches, size, ratios, csys, max_patches = DGLNetUtils.imgs_to_patches(org_imgs, patch_size)
                    ts_imgs, _ = self.patches_trans(patches, max_patches, csys, transform)
                    batch_size = ts_imgs.shape[0]
                    ts_imgs = self._to_var(ts_imgs)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs)
                # 全局与局部简单融合模式（拼接）
                elif self.args.mode_name.lower() == 'smpgl' or self.args.mode_name.lower() == 'cbamgl' \
                        or self.args.mode_name.lower() == 'attgl':
                    # 全局图像
                    glb_imgs = self.imgs_trans(org_imgs, transform)
                    # 将全局图分割成块儿
                    patch_size = (self.args.patch_size, self.args.patch_size)
                    patches, size, ratios, csys, max_patches = DGLNetUtils.imgs_to_patches(org_imgs, patch_size)
                    loc_imgs, _ = self.patches_trans(patches, max_patches, csys, transform)
                    batch_size, num, c, w, h = loc_imgs.size()
                    ts_imgs = torch.zeros(batch_size, num + 1, c, w, h)
                    # 将全局图与分块图合并到一个变量中
                    for j in range(len(org_imgs)):
                        # 第0个位置是全局图像
                        ts_imgs[j][0] = glb_imgs[j]
                        ts_imgs[j][1:] = loc_imgs[j]
                    ts_imgs = self._to_var(ts_imgs, requires_grad=False)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs)
                elif self.args.mode_name.lower() == 'cog2l' or self.args.mode_name.lower() == 'col2g':
                    # 全局图像
                    # 全局图像
                    glb_imgs = self.imgs_trans(org_imgs, transform)
                    # 将全局图分割成块儿
                    patch_size = (self.args.patch_size, self.args.patch_size)
                    patches, size, ratios, csys, max_patches = DGLNetUtils.imgs_to_patches(org_imgs, patch_size)
                    loc_imgs, csys = self.patches_trans(patches, max_patches, csys, transform)
                    csys = torch.tensor(csys).to(self.device)
                    ratios = torch.tensor(ratios).to(self.device)
                    batch_size, num, c, w, h = loc_imgs.size()
                    ts_imgs = torch.zeros(batch_size, num + 1, c, w, h)
                    # 将全局图与分块图合并到一个变量中
                    for j in range(len(org_imgs)):
                        # 第0个位置是全局图像
                        ts_imgs[j][0] = glb_imgs[j]
                        ts_imgs[j][1:] = loc_imgs[j]
                    ts_imgs = self._to_var(ts_imgs, requires_grad=False)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs, csys=csys, ratios=ratios)
                # 视觉特征抽取
                _, hidden_size, w, h = vis_enc_out.size()
                vis_enc_out = vis_enc_out.permute(0, 2, 3, 1).view(batch_size, -1, w, h, hidden_size)
                # -----------------------------------------------------------
                captions = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
                prob = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
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
        f_name = '{}_{}_{}_{}_{}'.format(self.now_time, self.args.model_name,
                                         self.mode, self.args.resize, self.args.batch_size)
        result_path = os.path.join(self.result_dir, 'report')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        rep_json = os.path.join(result_path, f_name + '.json')
        with open(rep_json, 'w') as f:
            json.dump(result, f)
            self.logger.info('The file {} has been saved.'.format(rep_json))
        return f_name


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Parallel', type=str, default='yes')
    parser.add_argument('--mode', type=str, default='train')
    # image size
    parser.add_argument('--media_type', type=str, default='jpg')
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--patch_size', type=int, default=384)
    parser.add_argument('--chan_attn', type=bool, default=False)
    parser.add_argument('--spat_attn', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--mode_name', type=str, default='col2g')

    parser.add_argument('--int_stop_dim', type=int, default=64)
    parser.add_argument('--sent_hidden_dim', type=int, default=512)
    parser.add_argument('--sent_input_dim', type=int, default=1024)
    parser.add_argument('--word_hidden_dim', type=int, default=512)
    parser.add_argument('--word_input_dim', type=int, default=512)

    parser.add_argument('--att_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--backward_step', type=int, default=12)
    parser.add_argument('--save_model_dir', type=str, default='results')
    parser.add_argument("--saved_model_name", type=str,
                        default='2021-06-30-01-54_resnet50_col2g_train_384_24_loss.pth.tar')

    args = parser.parse_args()
    genrep = DGLNetGenRep(dev, args)
    print('----------------------------------------------------')
    genrep.gen_reports(mode='train')
    genrep.gen_reports(mode='val')
    genrep.gen_reports(mode='test')
    # genrep.gen_eval_metrics(file_name)
