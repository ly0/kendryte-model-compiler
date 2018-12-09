from . import img_0_1


def load_dataset(args):
    dataset = img_0_1.load_dataset(args)
    return dataset * 2 - 1
