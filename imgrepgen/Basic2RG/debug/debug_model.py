import os
import torch
import argparse
from imgrepgen.Basic2RG.basic_trainer import BmirTrainer
from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--Parallel', type=str, default='no')
    parser.add_argument('--resize', type=int, default=224, help='size to which image is to be resized')
    parser.add_argument('--crop_size', type=int, default=224, help='size to which the image is to be cropped')
    parser.add_argument('--device_number', type=str, default="0", help='which GPU to run experiment on')
    parser.add_argument('--int_stop_dim', type=int, default=64,
                        help='intermediate state dimension of stop vector network')
    parser.add_argument('--sent_hidden_dim', type=int, default=512, help='hidden state dimension of sentence LSTM')
    parser.add_argument('--sent_input_dim', type=int, default=1024, help='dimension of input to sentence LSTM')
    parser.add_argument('--word_hidden_dim', type=int, default=512, help='hidden state dimension of word LSTM')
    parser.add_argument('--word_input_dim', type=int, default=512, help='dimension of input to word LSTM')
    parser.add_argument('--att_dim', type=int, default=64,
                        help='dimension of intermediate state in co-attention network')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in word LSTM')
    parser.add_argument('--lambda_sent', type=int, default=1,
                        help='weight for cross-entropy loss of stop vectors from sentence LSTM')
    parser.add_argument('--lambda_word', type=int, default=1,
                        help='weight for cross-entropy loss of words predicted from word LSTM with target words')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batch')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the instances in dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')
    parser.add_argument('--lr_rate_cnn', type=int, default=1e-5, help='learning rate for CNN Encoder')
    parser.add_argument('--lr_rate_lstm', type=int, default=5e-4, help='learning rate for LSTM Decoder')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--save_model_dir', type=str, default='../results', help='the path of the saved model')
    args = parser.parse_args()
    bmir_trainer = BmirTrainer(dev, args)
    bmir_trainer.train()




