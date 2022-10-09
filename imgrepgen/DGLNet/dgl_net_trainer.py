import argparse
import math
import sys
import torch.nn as nn
import torch
import os
from tqdm import tqdm
from time import strftime, localtime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datasets.nlmcxr.adv_big_loader as adv_big_loader
from imgrepgen.DGLNet.models.cogl_net import CoGLNet
from imgrepgen.DGLNet.models.glb_net import GlbNet
from medutils.imgutils.image_trans import ImageTrans
from medutils.logutils.logger_utils import MedLogger
from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader
from imgrepgen.DGLNet.models.lang_net import SentenceLSTM, WordLSTM
from imgrepgen.DGLNet.models.coglg_utils import DGLNetUtils


class DGLNetTrainer:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.model_dir = self._init_model_dir()
        self.now_time = self._get_now()
        self.logger = self._init_logger()
        self.vocab = adv_big_loader.ds_vocab(mode='train')
        self.load_batch_size = int(args.batch_size / args.backward_step)
        self.train_data_loader = self._init_data_loader(mode='train')
        self.val_data_loader = self._init_data_loader("val")
        # models
        self.vis_model, self.vis_params = self.__init_vis_models()
        self.vis_optimizer = torch.optim.Adam(params=self.vis_params, lr=self.args.vis_lr_rate)
        self.vis_scheduler = self._init_scheduler(self.vis_optimizer)

        self.sent_lstm, self.word_lstm, self.lang_params = self._init_lang_models()
        self.lang_optimizer = torch.optim.Adam(params=self.lang_params, lr=self.args.lang_lr_rate)
        self.lang_scheduler = self._init_scheduler(self.lang_optimizer)
        self.criterion_stop = self._init_cel_criterion()
        self.criterion_words = self._init_cel_criterion()
        self.criterion_stop_val = self._init_cel_criterion()
        self.criterion_words_val = self._init_cel_criterion()
        self.mini_train_loss = 10000000000
        self.mini_val_loss = 10000000000

    def _init_model_dir(self):
        root_dir = os.path.split(os.path.realpath(__file__))[0]
        model_dir = os.path.join(root_dir, self.args.save_model_dir)
        model_dir = os.path.join(model_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def _init_logger(self, mode='train'):
        log_file = "{}_{}_{}_{}_{}_loss.pth.tar".format(self.now_time,
                                                        self.args.model_name, self.args.mode_name.lower(), mode,
                                                        self.args.resize, self.args.batch_size)
        return MedLogger(log_file, self.model_dir).init_logger()

    def _init_cel_criterion(self):
        return nn.CrossEntropyLoss().to(self.device)

    # 调整学习率
    def _init_scheduler(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _to_var(self, x, requires_grad=True):
        x = x.to(self.device)
        x.requires_grad = requires_grad
        return x

    @staticmethod
    def _get_now():
        now_time = strftime('%Y-%m-%d-%H-%M', localtime())
        return now_time

    def __init_transform(self, mode='train'):
        tsf = ImageTrans(mode='train', resize=self.args.resize,
                         crop_size=self.args.crop_size).init_image_trans()
        return tsf

    def _init_data_loader(self, mode='train'):
        loader = adv_big_loader.dsLoader(mode=mode,
                                         media_type=self.args.media_type,
                                         batch_size=self.load_batch_size,
                                         s_max=10, n_max=50)
        return loader

    # 初始化视觉模型
    def __init_vis_models(self):
        vis_model = CoGLNet(self.args.model_name, self.args.chan_attn, self.args.spat_attn,
                            self.args.mode_name, self.args.patch_size)
        vis_model = self._set_to_dev(vis_model)
        vis_params = list(vis_model.parameters())
        return vis_model, vis_params

    def _init_lang_models(self):
        if self.args.Parallel == 'yes':
            enc_dim = self.vis_model.module.cogl_net.enc_dim
        else:
            enc_dim = self.vis_model.cogl_net.enc_dim

        # print('enc_dim size:', enc_dim)
        sent_lstm = SentenceLSTM(enc_dim, self.args.sent_hidden_dim,
                                 self.args.att_dim, self.args.sent_input_dim,
                                 self.args.word_input_dim, self.args.int_stop_dim)
        sent_lstm = self._set_to_dev(sent_lstm)
        word_lstm = WordLSTM(self.args.word_input_dim, self.args.word_hidden_dim, len(self.vocab), self.args.num_layers)
        word_lstm = self._set_to_dev(word_lstm)
        lang_params = list(sent_lstm.parameters()) + list(word_lstm.parameters())
        return sent_lstm, word_lstm, lang_params

    def _set_to_dev(self, model):
        if self.args.Parallel == 'yes':
            model = nn.DataParallel(model).to(self.device)
        else:
            model = model.to(self.device)
        return model

    def _save_model_params(self, mode='val', epoch=0, loss=1000):
        # DataParallel
        # torch.save(model.module.state_dict(), "model.pkl")
        def save_whole_model(_filename):
            if self.args.Parallel == 'yes':
                save_model = {'vis_model': self.vis_model.module.state_dict(),
                              'sent_lstm': self.sent_lstm.module.state_dict(),
                              'word_lstm': self.word_lstm.module.state_dict()
                              }
            else:
                save_model = {'vis_model': self.vis_model.state_dict(),
                              'sent_lstm': self.sent_lstm.state_dict(),
                              'word_lstm': self.word_lstm.state_dict()
                              }
            save_model['vis_optimizer'] = self.vis_optimizer.state_dict()
            save_model['lang_optimizer'] = self.lang_optimizer.state_dict()
            save_model['epoch'] = epoch
            torch.save(save_model, os.path.join(self.model_dir, "{}".format(_filename)))

        if loss < self.mini_train_loss:
            self.mini_train_loss = loss
            file_name = "{}_{}_{}_{}_{}_{}_loss.pth.tar".format(self.now_time, self.args.model_name,self.args.mode_name.lower(),
                                                             mode, self.args.resize, self.args.batch_size)
            self.logger.info("The saved model name is {}".format(file_name))
            save_whole_model(file_name)

    # 转换成tensor
    @staticmethod
    def imgs_trans(imgs, transform):
        # PIL images conver to tensor
        out_imgs = []
        for img in imgs:
            out_imgs.append(transform(img))
        out_imgs = torch.stack(out_imgs, dim=0)
        return out_imgs

    # 调整图像大小
    @staticmethod
    def img_resize(images, shape=None):
        resized = list(images)
        for i in range(len(images)):
            w, h = images[i].size
            # print('source image size:', w, h)
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

    # 将patches转换为tensor
    def patches_trans(self, patches, max_patches, csys, transform):
        loc_imgs = torch.zeros(self.load_batch_size, max_patches, 3, self.args.patch_size, self.args.patch_size)
        for j in range(len(patches)):
            ts_patches = self.imgs_trans(patches[j], transform)
            patch_num = ts_patches.size(0)
            loc_imgs[j][:patch_num] = ts_patches
            csys[j] += [[-1, -1]] * (max_patches - len(csys[j]))
        return loc_imgs, csys

    def train_model(self):
        mod_info = "Train model name:{}".format(self.args.mode_name)
        print(mod_info)
        self.logger.info(mod_info)
        for epoch in range(self.args.num_epochs):
            train_loss = self._epoch_train(epoch)
            val_loss = self._epoch_val(epoch)
            info = 'Training：Epoch [{}/{}],Train Loss:{}, Val Loss:{}'
            self.logger.info(info.format(epoch + 1, self.args.num_epochs, train_loss, val_loss))
            self.vis_scheduler.step(val_loss)
            self.lang_scheduler.step(val_loss)
            self._save_model_params('train', epoch, train_loss)

    def _epoch_train(self, epoch):
        self.vis_model.train()
        self.sent_lstm.train()
        self.word_lstm.train()
        train_loss = 0.0
        transform = self.__init_transform('train')
        train_step = len(self.train_data_loader)
        progress_bar = tqdm(self.train_data_loader, desc='Training')
        for i, (org_imgs, img_names, label, sentences, prob) in enumerate(progress_bar):
            # 调整图像大小
            org_imgs = self.img_resize(org_imgs)
            # org_imgs：是原始图像，不是tensor，在进行多GPU并行时，无法进进行数据分割，而是在每个GPU上复制一份
            vis_enc_out = None
            batch_size = None
            # glb:全局网络模式
            if self.args.mode_name.lower() == 'glb':
                ts_imgs = self.imgs_trans(org_imgs, transform)
                batch_size, c, w, h = ts_imgs.size()
                ts_imgs = ts_imgs.view(batch_size, -1, c, w, h)
                ts_imgs = self._to_var(ts_imgs)
                vis_enc_out = self.vis_model.forward(images=ts_imgs)
            # loc:局部网络模式
            elif self.args.mode_name.lower() == 'loc':
                # 将一个原始图像分割成块(imgs_in->patches)
                patch_size = (self.args.patch_size, self.args.patch_size)
                patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
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
                patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
                loc_imgs, _ = self.patches_trans(patches, max_patches, csys, transform)
                batch_size, num, c, w, h = loc_imgs.size()
                ts_imgs = torch.zeros(batch_size, num + 1, c, w, h)
                # 将全局图与分块图合并到一个变量中
                for j in range(len(org_imgs)):
                    # 第0个位置是全局图像
                    ts_imgs[j][0] = glb_imgs[j]
                    ts_imgs[j][1:] = loc_imgs[j]
                ts_imgs = self._to_var(ts_imgs)
                vis_enc_out = self.vis_model.forward(images=ts_imgs)
            elif self.args.mode_name.lower() == 'cog2l' \
                    or self.args.mode_name.lower() == 'col2g' \
                    or self.args.mode_name.lower() == 'coglg':
                # 全局图像
                glb_imgs = self.imgs_trans(org_imgs, transform)
                # 将全局图分割成块儿
                patch_size = (self.args.patch_size, self.args.patch_size)
                patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
                loc_imgs, csys = self.patches_trans(patches, max_patches, csys,transform)
                csys = torch.tensor(csys).to(self.device)
                ratios = torch.tensor(ratios).to(self.device)
                batch_size, num, c, w, h = loc_imgs.size()
                ts_imgs = torch.zeros(batch_size, num + 1, c, w, h)
                # 将全局图与分块图合并到一个变量中
                for j in range(len(org_imgs)):
                    # 第0个位置是全局图像
                    ts_imgs[j][0] = glb_imgs[j]
                    ts_imgs[j][1:] = loc_imgs[j]
                ts_imgs = self._to_var(ts_imgs)
                vis_enc_out = self.vis_model.forward(images=ts_imgs, csys=csys, ratios=ratios)
            elif self.args.mode_name.lower() == 'dglnet':
                # 全局图像
                glb_imgs = self.imgs_trans(org_imgs, transform)
                # 将全局图分割成块儿
                patch_size = (self.args.patch_size, self.args.patch_size)
                patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
                loc_imgs, csys = self.patches_trans(patches, max_patches, csys,transform)
                csys = torch.tensor(csys).to(self.device)
                ratios = torch.tensor(ratios).to(self.device)
                batch_size, num, c, w, h = loc_imgs.size()
                ts_imgs = torch.zeros(batch_size, num + 1, c, w, h)
                # 将全局图与分块图合并到一个变量中
                for j in range(len(org_imgs)):
                    # 第0个位置是全局图像
                    ts_imgs[j][0] = glb_imgs[j]
                    ts_imgs[j][1:] = loc_imgs[j]
                ts_imgs = self._to_var(ts_imgs)
                vis_enc_out = self.vis_model.forward(images=ts_imgs, csys=csys, ratios=ratios)

            print("vis enc out:", vis_enc_out.size())
            # vis_enc_out [batch_size,hidden_size,12,12]
            # 视觉特征抽取
            _,  hidden_size, w, h = vis_enc_out.size()
            # vis_enc_out = vis_enc_out.permute(0, 2, 3, 1)
            vis_enc_out = vis_enc_out.view(batch_size, -1, w, h, hidden_size)
            # ------------------------------------------------------------------------------
            sentences = self._to_var(torch.Tensor(sentences).long(), requires_grad=False)
            prob = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
            topics, stop_p = self.sent_lstm(vis_enc_out, sentences, self.device)
            # topics[2,9,512], ps[2,2]
            print("stop_p {}".format(stop_p.view(-1, 2).size()))
            print("prob.view {}".format(prob.view(-1).size()))
            loss_sent = self.criterion_stop(stop_p.view(-1, 2), prob.view(-1))
            loss_word = torch.tensor([0.0]).to(self.device)
            sent_num = sentences.shape[1]
            for sent_index in range(sent_num):
                word_outputs = self.word_lstm(vis_enc_out, topics[:, sent_index, :], sentences[:, sent_index, :])

                curr_word_loss = self.criterion_words(word_outputs.contiguous().view(-1, len(self.vocab)),
                                                      sentences[:, sent_index, :].contiguous().view(-1))
                
                loss_word += curr_word_loss
            batch_loss = self.args.lambda_sent * loss_sent + self.args.lambda_word * loss_word
            batch_loss.backward()

            if (i + 1) % self.args.backward_step == 0:
                self.vis_optimizer.step()
                self.vis_optimizer.zero_grad()
                self.lang_optimizer.step()
                self.lang_optimizer.zero_grad()
            info = 'Training: Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.6f}'.format(epoch, self.args.num_epochs, i,
                                                                                      train_step, batch_loss.item())
            progress_bar.set_description(info)
            train_loss += batch_loss.item()
            if math.isnan(train_loss):
                print('The batch_loss is null,the system is out.')
                sys.exit()
            # 清楚变量缓存
            # del org_imgs, ts_imgs, vis_enc_out, captions, batch_loss, prob, loss_sent, loss_word
            # torch.cuda.empty_cache()
        avg_train_loss = train_loss / train_step
        print('train loss:{}'.format(avg_train_loss))
        return avg_train_loss

    def _epoch_val(self, epoch):
        self.vis_model.eval()
        self.sent_lstm.eval()
        self.word_lstm.eval()
        val_loss = 0
        transform = self.__init_transform('eval')
        val_step = len(self.val_data_loader)
        progress_bar = tqdm(self.val_data_loader, desc='Evaluating')
        with torch.no_grad():
            for i, (org_imgs, img_names, label, captions, prob) in enumerate(progress_bar):
                # 调整图像大小
                org_imgs = self.img_resize(org_imgs)
                # org_imgs：是原始图像，不是tensor，在进行多GPU并行时，无法进进行数据分割，而是在每个GPU上复制一份
                vis_enc_out = None
                batch_size = None
                # glb:全局网络模式
                if self.args.mode_name.lower() == 'glb':
                    ts_imgs = self.imgs_trans(org_imgs, transform)
                    batch_size, c, w, h = ts_imgs.size()
                    ts_imgs = ts_imgs.view(batch_size, -1, c, w, h)
                    ts_imgs = self._to_var(ts_imgs,requires_grad=False)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs)
                # loc:局部网络模式
                elif self.args.mode_name.lower() == 'loc':
                    # 将一个原始图像分割成块(imgs_in->patches)
                    patch_size = (self.args.patch_size, self.args.patch_size)
                    patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
                    ts_imgs, _ = self.patches_trans(patches, max_patches, csys, transform)
                    batch_size = ts_imgs.shape[0]
                    ts_imgs = self._to_var(ts_imgs,requires_grad=False)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs)
                # 全局与局部简单融合模式（拼接）
                elif self.args.mode_name.lower() == 'smpgl' or self.args.mode_name.lower() == 'cbamgl' \
                        or self.args.mode_name.lower() == 'attgl':
                    # 全局图像
                    glb_imgs = self.imgs_trans(org_imgs, transform)
                    # 将全局图分割成块儿
                    patch_size = (self.args.patch_size, self.args.patch_size)
                    patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
                    loc_imgs, _ = self.patches_trans(patches, max_patches, csys, transform)
                    batch_size, num, c, w, h = loc_imgs.size()
                    ts_imgs = torch.zeros(batch_size, num + 1, c, w, h)
                    # 将全局图与分块图合并到一个变量中
                    for j in range(len(org_imgs)):
                        # 第0个位置是全局图像
                        ts_imgs[j][0] = glb_imgs[j]
                        ts_imgs[j][1:] = loc_imgs[j]
                    ts_imgs = self._to_var(ts_imgs,requires_grad=False)
                    vis_enc_out = self.vis_model.forward(images=ts_imgs)
                elif self.args.mode_name.lower() == 'cog2l' \
                        or self.args.mode_name.lower() == 'col2g' \
                        or self.args.mode_name.lower() == 'coglg':
                    # 全局图像
                    glb_imgs = self.imgs_trans(org_imgs, transform)
                    # 将全局图分割成块儿
                    patch_size = (self.args.patch_size, self.args.patch_size)
                    patches, size, ratios, csys, max_patches = DGLNetUtils.glb_imgs_to_patches(org_imgs, patch_size)
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
                    ts_imgs = self._to_var(ts_imgs,requires_grad=False)

                    vis_enc_out = self.vis_model.forward(images=ts_imgs, csys=csys, ratios=ratios)

                print("vis enc out:", vis_enc_out.size())
                # 视觉特征抽取
                _, w, h, hidden_size = vis_enc_out.size()
                # vis_enc_out = vis_enc_out.permute(0, 2, 3, 1)
                vis_enc_out = vis_enc_out.view(batch_size, -1, w, h, hidden_size)
                captions = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
                prob = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

                topics, ps = self.sent_lstm(vis_enc_out, captions, self.device)

                loss_sent = self.criterion_stop_val(ps.view(-1, 2), prob.view(-1))

                loss_word = torch.tensor([0.0]).to(self.device)
                pred_words = torch.zeros((captions.shape[0], captions.shape[1], captions.shape[2]))
                # 循环句子
                for sent_index in range(captions.shape[1]):

                    word_outputs = self.word_lstm(topics[:, sent_index, :], captions[:, sent_index, :])

                    loss_word += self.criterion_words_val(word_outputs.contiguous().view(-1, len(self.vocab)),
                                                          captions[:, sent_index, :].contiguous().view(-1))
                    _, words = torch.max(word_outputs, 2)
                    pred_words[:, sent_index, :] = words
                batch_loss = self.args.lambda_sent * loss_sent + self.args.lambda_word * loss_word
                info = 'Evaluating: Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.6f}'.format(epoch, self.args.num_epochs,
                                                                                            i,
                                                                                            val_step,
                                                                                            batch_loss.item())
                progress_bar.set_description(info)
                val_loss += batch_loss.item()
                # 清楚变量缓存
                del org_imgs, captions, batch_loss, prob, loss_sent, loss_word
                torch.cuda.empty_cache()
            avg_val_loss = val_loss / val_step
        return avg_val_loss


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--Parallel', type=str, default='yes')
    parser.add_argument('--model_name', type=str, default='resnet50')
    # parser.add_argument('--mode_name', type=str, default='cbamgl')
    # image size
    parser.add_argument('--media_type', type=str, default='jpg')
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--patch_size', type=int, default=384)
    parser.add_argument('--chan_attn', type=bool, default=False)
    parser.add_argument('--spat_attn', type=bool, default=False)
    # 实际上每个patch加载，24/12=2个图像,但每12个patch优化一次，step一次
    # parser.add_argument('--batch_size', type=int, default=24)
    # parser.add_argument('--backward_step', type=int, default=12)
    parser.add_argument('--int_stop_dim', type=int, default=64)
    parser.add_argument('--sent_hidden_dim', type=int, default=512)
    parser.add_argument('--sent_input_dim', type=int, default=1024)
    parser.add_argument('--word_hidden_dim', type=int, default=512)
    parser.add_argument('--word_input_dim', type=int, default=512)
    parser.add_argument('--att_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in word LSTM')
    parser.add_argument('--lambda_sent', type=int, default=1)
    parser.add_argument('--lambda_word', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--vis_lr_rate', type=int, default=1e-5)
    parser.add_argument('--lang_lr_rate', type=int, default=5e-4)
    parser.add_argument('--save_model_dir', type=str, default='results')
    # self.mode = mode
    print('-----------------------------------------------------------')
    print('CoGL：基于全局与局部特征协同融合的影像报告生成模型训练')
    print('请输入下列训练模式中的一种：')
    print('1. 全局特征模式(经典模式)：Glb')
    print('2. 局部特征模式(图像块儿)：Loc')
    print('3. 全局与局部特征简单融合模式：SmpGL')
    print('3. 全局与局部特征自注意力融合模式：ATTGL')
    print('4. 全局与局部特征CBAM注意力融合模式：CbamGL')
    print('5. 全局到局部协特征同融合模式：CoG2L')
    print('6. 局部到全局特征协同融合模式：CoL2G')
    print('7. 全局与全局特征双向协同融合模式：DGLNet')
    print('-----------------------------------------------------------')
    mode_name = input()
    parser.add_argument('--mode_name', type=str, default=mode_name)
    b_size = input('请输入批处理数量：')
    parser.add_argument('--batch_size', type=int, default=b_size)
    b_step = input('请输入反向传播的步数：')
    parser.add_argument('--backward_step', type=int, default=b_step)
    pargs = parser.parse_args()
    trainer = DGLNetTrainer(dev, pargs)
    trainer.train_model()
