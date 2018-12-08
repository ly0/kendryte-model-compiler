import sys
import os

def signed_to_hex(value, width):
    return hex(int(round((1 << width) + value)) % (1 << width))


def debug_format_line(line, fout):
    line = [*line, *([0] * (64 - len(line)))]
    ret = ''.join([format(v, '02x') + ('  ' if i % 8 == 7 else ('--' if i % 8 == 3 else '')) for v, i in
                   zip(line, range(len(line)))])
    fout.write('Address 0X00000000: ' + ret + '\n')


def split_chunks(L, n):
    for i in range(0, len(L), n):
        yield L[i:i + n]


def min_max_to_scale_bias(minv, maxv):
    scale = (maxv - minv) / 255
    bias = minv
    return scale, bias

def import_from_path(module_path):
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path)

    if module_name.endswith('.py'):
        module_name = module_name[:-3]

    sys_path = sys.path
    sys.path = [os.path.abspath(module_dir)] if module_dir else ['.']
    loaded_module = __import__(module_name)
    sys.path = sys_path

    return loaded_module