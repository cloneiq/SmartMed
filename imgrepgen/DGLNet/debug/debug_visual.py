import torch
from torchvision import transforms
from ..models.glb_net import GlbNet
from imgrepgen.DGLNet.models.cogl_net import CoGLNet
transformer = transforms.Compose([
    transforms.ToTensor()
])


def images_transform(images):
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0)
    return inputs


def resize(images, shape):
    resized = list(images)
    for i in range(len(images)):
        resized[i] = images[i].resize(shape)
    return resized


if __name__ == "__main__":
    # 初始化两个图像
    img1 = torch.randn(3, 2048, 2048)
    img1 = transforms.ToPILImage()(img1)
    img2 = torch.randn(3, 2048, 2048)
    img2 = transforms.ToPILImage()(img2)
    imgs = [img1, img2]
    glb_imgs = resize(imgs, (512, 512))  # list of resized PIL images
    glb_imgs = images_transform(glb_imgs)  # 转换成tensor
    # 全局网络
    glb_net = GlbNet()
    glb_net.forward(glb_imgs)





