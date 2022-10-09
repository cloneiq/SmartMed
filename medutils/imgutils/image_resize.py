import argparse
import os
from PIL import Image
from tqdm import tqdm


def resize(images, shape, quality=Image.BILINEAR):
    resized = list(images)
    for i in range(len(images)):
        resized[i] = images[i].resize(shape, quality)
    return resized


def resize_image(image, shape, quality=Image.BILINEAR):
    return image.resize(shape, quality)


def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = os.listdir(image_dir)
    num_images = len(images)
    prog_bar = tqdm(images, desc='Transforming')
    for i, image in enumerate(prog_bar):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        info = "[{}/{}] Resized the images and saved into '{}'." \
            .format(i + 1, num_images, output_dir)
        prog_bar.set_description(info)


def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='F:/NLMCXR_JPGs',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='D:/data/coco/imgs_in/val2014',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
