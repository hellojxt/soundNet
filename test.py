import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train envelope net or frequency net.')
    parser.add_argument('--res', type=int, default=8,help='resolution for frequency')
    parser.add_argument('--net', type=str, help='class name of the network')
    parser.add_argument('--epoch', type=int, help='max epoch')
    args = parser.parse_args()
    print(args.res)