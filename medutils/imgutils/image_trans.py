from torchvision import transforms
import torch


class ImageTrans:
    def __init__(self, mode='train', resize=256, crop_size=224):
        self.mode = mode
        self.resize = resize
        self.crop_size = crop_size

    def init_image_trans(self):
        image_transforms = {
            'train': transforms.Compose([
                # 将图像进行缩放，缩放为
                transforms.Resize(self.resize),
                transforms.RandomCrop(self.crop_size),
                # 在256*256的图像上随机裁剪出224*224大小的图像用于训练
                # transforms.RandomResizedCrop(224),
                # 图像用于翻转
                transforms.RandomHorizontalFlip(),
                # 转换成tensor向量
                transforms.ToTensor(),
                # 对图像进行归一化操作
                # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224，不需要翻转
            'val': transforms.Compose([
                transforms.Resize((self.crop_size, self.crop_size)),
                # transforms.RandomCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((self.crop_size, self.crop_size)),
                # transforms.RandomCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        return image_transforms[self.mode]

    # 转换成tensor
    def image_trans(self, imgs):
        # PIL images conver to tensor
        transform = self.init_image_trans()
        out_imgs = []
        for img in imgs:
            out_imgs.append(transform(img))
        out_imgs = torch.stack(out_imgs, dim=0)
        return out_imgs
