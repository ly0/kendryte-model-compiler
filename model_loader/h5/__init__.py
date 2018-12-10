import os

def load_model(dataset, range_from_batch, args):
    import tools
    pb_loader = tools.import_from_path(os.path.dirname(__file__) + '/../pb')
    from . import h5_converter
    if not args.pb_path.endswith('.h5'):
        raise ValueError('{} should endwith *.h5'.format(args.pb_path))

    args.pb_path = h5_converter.convert(args.pb_path)

    return pb_loader.load_model(dataset, range_from_batch, args)

