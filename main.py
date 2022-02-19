import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description='BLeafNet: A Bonferroni mean operator based fusion of CNN models for plant identification using leaf image classification')

    parser.add_argument('--train', type=str, nargs = '+', help = 'What is the path of the training leaf image data?')
    parser.add_argument('--val', type=str, nargs = '+', help = 'What is the path of the Validation leaf image data?')
    parser.add_argument('--epochs_base', default=60, type=int, help = 'What is the training epoch for base CNN models?')
    parser.add_argument('--epochs', type=int, default=150, help = 'What is the training epoch for fused model?')
    parser.add_argument('--batch', type=int, default=32, help = 'What is the training batch size?')
    parser.add_argument('--steps', type=int, default=40, help = 'What is the training steps per epoch?')

    args = parser.parse_args()
    args.train = ' '.join(args.train)
    args.val = ' '.join(args.val)

    print("Configuration")
    print("----------------------------------------------------------------------")
    print("Training Path : {}".format(args.train))
    print("Validation Path : {}".format(args.val))
    print("Epochs while training base CNN model: {}".format(args.epochs_base))
    print("Epochs while training Fusion model : {}".format(args.epochs))
    print("Batch Size : {}".format(args.batch))
    print("Steps per epochs : {}".format(args.steps))
    print("----------------------------------------------------------------------")

    train(args.train, args.val).run(args.epochs_base, args.epochs, args.batch, args.steps)

if __name__=='__main__':
    main()
