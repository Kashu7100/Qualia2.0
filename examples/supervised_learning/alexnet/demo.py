from qualia2.vision import Alexnet, imagenet_labels
import qualia2.vision.transforms as transforms
import PIL
import numpy
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRU example with Qualia2.0')
    parser.add_argument('image', metavar='str', type=str, help='path to an image.')
    parser.add_argument('-n', '--top_n', type=int, default=5, help=' Number of candidates to show. Default: 5')

    args = parser.parse_args()

    img = PIL.Image.open(args.image)

    preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize()
    ])

    input = preprocess(img)

    model = Alexnet(pretrained=True)
    model.eval()

    output = model(input).asnumpy()
    sorted = output.argsort()[:,-args.top_n:][:,::-1]

    for i, candidates in enumerate(sorted):
       for idx in candidates:
          print('{}: {:.2f}%'.format(imagenet_labels[idx], output[i,idx]*100))
