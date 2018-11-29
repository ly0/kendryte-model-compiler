
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
