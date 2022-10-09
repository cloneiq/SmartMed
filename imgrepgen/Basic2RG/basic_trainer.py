# import time
from time import strftime, localtime
import torch
import argparse
import os
import torch.nn as nn
from torchvision import transforms
import datasets.nlmcxr.adv_loader as advLoader
from imgrepgen.Basic2RG.models.basic_models import SentenceLSTM, WordLSTM
import modelfactory.vision.cnn_models as cnn_models
from medutils.imgutils.image_trans import ImageTrans
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from medutils.logutils.logger_utils import MedLogger
from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader


class BasicTrainer:
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.model_dir = self._init_model_dir()
        self.now_time = self._get_now()
        self.logger = self._init_logger()
        self.vocab = advLoader.ds_vocab(mode='train')
        self.train_data_loader = self._init_data_loader('train')
        self.val_data_loader = self._init_data_loader("val")
        # Initialize the CNN model
        self.cnn_pretrained = self.args.cnn_model_pretrained
        self.cnn_model, self.cnn_params = self._init_cnn_models()
        if not self.cnn_pretrained:
            self.optimizer_cnn = self._init_optimizer(self.cnn_params, self.args.lr_rate_cnn)
            self.cnn_scheduler = self._init_scheduler(self.optimizer_cnn)
        # Initialize the LSTM model
        self.sent_lstm, self.word_lstm, self.lstm_params = self._init_lstm_models()
        self.optimizer_lstm = self._init_optimizer(self.lstm_params, self.args.lr_rate_lstm)
        self.lstm_scheduler = self._init_scheduler(self.optimizer_lstm)
        #
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
                                                        self.args.cnn_model_name, mode,
                                                        self.args.resize, self.args.batch_size)
        return MedLogger(log_file, self.model_dir).init_logger()

    def _init_data_loader(self, mode='train'):
        tfs = ImageTrans(mode=mode, resize=self.args.resize,
                         crop_size=self.args.crop_size).init_image_trans()
        loader = advLoader.dsLoader(mode=mode,
                                    transform=tfs,
                                    media_type='jpg',
                                    batch_size=self.args.batch_size,
                                    s_max=10, n_max=50)
        return loader

    def _init_cnn_models(self):
        cnn_model = cnn_models.create_model(self.args.cnn_model_name, pretrained=self.cnn_pretrained,
                                            chan_attn=self.args.chan_attn, spat_attn=self.args.spat_attn)
        cnn_model = self._setModelToDev(cnn_model)
        if self.cnn_pretrained:
            self.logger.info("Using the cnn pretrained model....")
            for param in cnn_model.parameters():
                param.requires_grad = False
        cnn_params = list(cnn_model.parameters())
        return cnn_model, cnn_params

    def _init_lstm_models(self):
        if self.args.Parallel == 'yes':
            enc_dim = self.cnn_model.module.enc_dim
        else:
            enc_dim = self.cnn_model.enc_dim
        sent_lstm = SentenceLSTM(enc_dim, self.args.sent_hidden_dim,
                                 self.args.att_dim, self.args.sent_input_dim,
                                 self.args.word_input_dim, self.args.int_stop_dim)
        sent_lstm = self._setModelToDev(sent_lstm)
        word_lstm = WordLSTM(self.args.word_input_dim, self.args.word_hidden_dim, len(self.vocab), self.args.num_layers)
        word_lstm = self._setModelToDev(word_lstm)
        lstm_params = list(sent_lstm.parameters()) + list(word_lstm.parameters())
        return sent_lstm, word_lstm, lstm_params

    def _setModelToDev(self, model):
        if self.args.Parallel == 'yes':
            model = nn.DataParallel(model).to(self.device)
        else:
            model = model.to(self.device)
        return model

    def _init_cel_criterion(self):
        return nn.CrossEntropyLoss().to(self.device)

    def _init_msel_criterion(self):
        return nn.MSELoss().to(self.device)

    @staticmethod
    def _init_optimizer(params, lr):
        return torch.optim.Adam(params=params, lr=lr)

    # 调整学习率
    def _init_scheduler(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _to_var(self, x, requires_grad=True):
        x = x.to(self.device)
        x.requires_grad = requires_grad
        return x

    def _save_model_params(self, mode='val', epoch=0, loss=1000):
        # DataParallel
        # torch.save(model.module.state_dict(), "model.pkl")
        def save_whole_model(_filename):
            if self.args.Parallel == 'yes':
                save_model = {'cnn_model': self.cnn_model.module.state_dict(),
                              'sent_lstm': self.sent_lstm.module.state_dict(),
                              'word_lstm': self.word_lstm.module.state_dict()
                              }
            else:
                save_model = {'cnn_model': self.cnn_model.state_dict(),
                              'sent_lstm': self.sent_lstm.state_dict(),
                              'word_lstm': self.word_lstm.state_dict()
                              }
            if not self.cnn_pretrained:
                save_model['optimizer_cnn'] = self.optimizer_cnn.state_dict()
            save_model['optimizer_lstm'] = self.optimizer_lstm.state_dict()
            save_model['epoch'] = epoch
            torch.save(save_model, os.path.join(self.model_dir, "{}".format(_filename)))

        if loss < self.mini_train_loss:
            self.mini_train_loss = loss
            file_name = "{}_{}_{}_{}_{}_loss.pth.tar".format(self.now_time, self.args.cnn_model_name,
                                                             mode, self.args.resize, self.args.batch_size)
            self.logger.info("The saved model name is {}".format(file_name))
            save_whole_model(file_name)

    @staticmethod
    def _get_now():
        now_time = strftime('%Y-%m-%d-%H-%M', localtime())
        return now_time

    def train_model(self):
        for epoch in range(self.args.num_epochs):
            train_loss = self._epoch_train(epoch)
            val_loss = self._epoch_val(epoch)
            info = 'Training：Epoch [{}/{}],Train Loss:{}, Val Loss:{}'
            self.logger.info(info.format(epoch + 1, self.args.num_epochs, train_loss, val_loss))
            if not self.cnn_pretrained:
                self.cnn_scheduler.step(val_loss)
            self.lstm_scheduler.step(val_loss)
            self._save_model_params('train', epoch, train_loss)

    # 读取训练集数据进行模型训练
    def _epoch_train(self, epoch):
        self.cnn_model.train()
        self.sent_lstm.train()
        self.word_lstm.train()
        train_loss = 0
        train_step = len(self.train_data_loader)
        progress_bar = tqdm(self.train_data_loader, desc='Training')
        for i, (images, img_names, label, captions, prob) in enumerate(progress_bar):
            if not self.cnn_pretrained:
                images = self._to_var(images)
            else:
                images = self._to_var(images, requires_grad=False)
            vis_enc_out = self.cnn_model(images).permute(0, 2, 3, 1)
            # batch_size = label.size(0)
            # _, hidden_size, w, h = vis_enc_out.size()
            # vis_enc_out = vis_enc_out.permute(0, 2, 3, 1).view(batch_size, -1, w, h, hidden_size)
            captions = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
            # print('vis_ enc output---------',vis_enc_out.size())
            # vis_enc_output size： torch.Size([8, 7, 7, 512])
            topics, ps = self.sent_lstm(vis_enc_out, captions, self.device)
            # topics size: torch.Size([8, 11, 512]) ps size:torch.Size([8, 11, 2])
            loss_sent = self.criterion_stop(ps.view(-1, 2), prob.view(-1))
            loss_word = torch.tensor([0.0]).to(self.device)
            # 循环句子
            for sent_index in range(captions.shape[1]):
                word_outputs = self.word_lstm(topics[:, sent_index, :], captions[:, sent_index, :])
                loss_word += self.criterion_words(word_outputs.contiguous().view(-1, len(self.vocab)),
                                                  captions[:, sent_index, :].contiguous().view(-1))

            batch_loss = self.args.lambda_sent * loss_sent + self.args.lambda_word * loss_word
            batch_loss.backward()

            if (i + 1) % self.args.backward_step == 0:
                if not self.cnn_pretrained:
                    self.optimizer_cnn.step()
                    self.optimizer_cnn.zero_grad()
                self.optimizer_lstm.step()
                self.optimizer_lstm.zero_grad()

            # if i % self.args.log_step == 0:
            info = 'Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(epoch, self.args.num_epochs, i,
                                                                                train_step, batch_loss.item())
            progress_bar.set_description(info)
            train_loss += batch_loss.item()
            # 清楚变量缓存
            del images, captions, batch_loss, prob, loss_sent, loss_word
            torch.cuda.empty_cache()
        avg_train_loss = train_loss / train_step
        return avg_train_loss

    # 读取验证集进行验证
    def _epoch_val(self, epoch):
        self.cnn_model.eval()
        self.sent_lstm.eval()
        self.word_lstm.eval()
        val_loss = 0
        val_step = len(self.val_data_loader)
        progress_bar = tqdm(self.val_data_loader, desc='Evaluating')
        with torch.no_grad():
            for i, (images, img_names, label, captions, prob) in enumerate(progress_bar):
                # print('images size:-----',images.size())
                images = self._to_var(images, requires_grad=False)
                vis_enc_out = self.cnn_model(images).permute(0, 2, 3, 1)
                # batch_size = label.size(0)
                # _, hidden_size, w, h = vis_enc_out.size()
                # vis_enc_out = vis_enc_out.permute(0, 2, 3, 1).view(batch_size, -1, w, h, hidden_size)
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

                # if i % self.args.log_step == 0:
                #     print('\nEpoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                #           .format(epoch, self.args.num_epochs, i, val_step, round(batch_loss.item(), 6)))
                info = 'Evaluating: Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(epoch, self.args.num_epochs, i,
                                                                                      val_step,
                                                                                      batch_loss.item())
                progress_bar.set_description(info)
                val_loss += batch_loss.item()

                # 清楚变量缓存
                del images, captions, batch_loss, prob, loss_sent, loss_word
                torch.cuda.empty_cache()
            avg_val_loss = val_loss / val_step
        return avg_val_loss


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--Parallel', type=str, default='yes')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--cnn_model_name', type=str, default='resnet50')
    parser.add_argument('--cnn_model_pretrained', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--backward_step', type=int, default=1)
    parser.add_argument('--chan_attn', type=bool, default=True)
    parser.add_argument('--spat_attn', type=bool, default=False)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--int_stop_dim', type=int, default=64)
    parser.add_argument('--sent_hidden_dim', type=int, default=512)
    parser.add_argument('--sent_input_dim', type=int, default=1024)
    parser.add_argument('--word_hidden_dim', type=int, default=512)
    parser.add_argument('--word_input_dim', type=int, default=512)
    parser.add_argument('--att_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in word LSTM')
    parser.add_argument('--lambda_sent', type=int, default=1)
    parser.add_argument('--lambda_word', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr_rate_cnn', type=int, default=1e-5)
    parser.add_argument('--lr_rate_lstm', type=int, default=5e-4)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_model_dir', type=str, default='results')
    pargs = parser.parse_args()
    pargs.batch_size = int(pargs.batch_size / pargs.backward_step)
    bmir_trainer = BasicTrainer(dev, pargs)
    bmir_trainer.train_model()
