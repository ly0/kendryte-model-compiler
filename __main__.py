'''
 * Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import argparse
import os
import sys
import tempfile

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile

import range_from_batch
import tensor_head_to_tensor_list
import tensor_list_to_layer_list
import layer_list_to_k210_layer
import k210_layer_to_c_code
import k210_layer_to_bin
import tools

current_dir = os.path.dirname(os.path.realpath(__file__))


def load_graph(pb_file_path, tensor_output_name, tensor_input_name):
    if pb_file_path.endswith('h5'):
        import h5_converter
        pb_file_path = h5_converter.convert(pb_file_path)

    if pb_file_path.endswith('pb'):
        with tf.Session() as persisted_sess:
            with gfile.GFile(pb_file_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

        output_tensor, input_tensor = None, None
        if tensor_output_name is not None:
            output_tensor = persisted_sess.graph._nodes_by_name[tensor_output_name].outputs[0]
        if tensor_input_name is not None:
            input_tensor = persisted_sess.graph._nodes_by_name[tensor_input_name].outputs[0]

        return output_tensor, input_tensor

    return None


def overwride_is_training_name(dataset, name):
    with tf.Session() as sess:
        try:
            is_training = sess.graph.get_operation_by_name(name)
            if is_training is not None:
                dataset[name + ':0'] = False
        except:
            pass

    return dataset


def overwride_is_training(dataset):
    dataset = overwride_is_training_name(dataset, 'is_training')
    dataset = overwride_is_training_name(dataset, 'phase_train')
    return dataset


def convert(tensor_output, tensor_input, dataset, eight_bit_mode=False, input_minmax_auto=False, input_min=0,
            input_max=1, prefix='', layer_start_idx=0):
    with tf.Session() as sess:
        converter = tensor_head_to_tensor_list.PbConverter(tensor_output, tensor_input)
        converter.convert()
        layers = tensor_list_to_layer_list.convert_to_layers(sess, dataset, converter.dst)

        rfb = range_from_batch.RangeFromBatchMinMax()
        if input_minmax_auto:
            input_min, input_max, = rfb(sess, tensor_input, dataset)
            in_scale, in_bias = tools.min_max_to_scale_bias(input_min, input_max)
            print('[layer input] scale/bias:', in_scale, in_bias)

        k210_layers = layer_list_to_k210_layer.gen_k210_layers(
            layers, sess, dataset,
            range_from_batch=rfb,
            eight_bit_mode=eight_bit_mode,
            input_min=input_min,
            input_max=input_max,
            layer_start_idx=layer_start_idx
        )

        output_code = k210_layer_to_c_code.gen_layer_list_code(k210_layers, eight_bit_mode, prefix, layer_start_idx)
        try:
            output_bin = k210_layer_to_bin.gen_layer_bin(k210_layers, eight_bit_mode)
        except Exception as e:
            print(e)
            output_bin = None

        return (output_code, output_bin)


def main():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard_mode', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pb_path', type=str, default='<please set --pb_path>', required=True)
    parser.add_argument('--tensor_input_name', default=None)
    parser.add_argument('--tensor_output_name', default=None)
    parser.add_argument('--tensor_input_min', type=float, default=0)
    parser.add_argument('--tensor_input_max', type=float, default=1)
    parser.add_argument('--tensor_input_minmax_auto', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--eight_bit_mode', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--output_path', default='build/gencode_output')
    parser.add_argument('--output_bin_name', default='build/model.bin')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--layer_start_idx', type=int, default=0)

    parser.add_argument('--dataset_loader', default='img_loader')
    parser.add_argument('--dataset_input_name', default='input:0')
    parser.add_argument('--dataset_pic_path', default='dataset/yolo')
    parser.add_argument('--image_w', type=int, default=320)
    parser.add_argument('--image_h', type=int, default=240)

    args = parser.parse_args()

    tensorboard_mode = args.tensorboard_mode
    pb_path = args.pb_path
    tensor_input_name = args.tensor_input_name
    tensor_output_name = args.tensor_output_name
    input_min = args.tensor_input_min
    input_max = args.tensor_input_max
    input_minmax_auto = args.tensor_input_minmax_auto
    eight_bit_mode = args.eight_bit_mode
    output_path = args.output_path
    output_bin_name = args.output_bin_name
    prefix = args.prefix if len(args.prefix) > 0 \
        else os.path.basename(args.output_path).replace('.', '_').replace('-', '_')

    layer_start_idx = args.layer_start_idx

    dataset_input_name = args.dataset_input_name
    dataset_pic_path = args.dataset_pic_path  # used in dataset loader
    image_w = args.image_w  # used in dataset loader
    image_h = args.image_h  # used in dataset loader
    dataset_loader = args.dataset_loader

    if ':' not in dataset_input_name:
        dataset_input_name = dataset_input_name + ':0'

    if output_path.endswith('.c'):
        output_path = output_path[:-2]

    if tensorboard_mode:
        load_graph(pb_path, None, None)
        graphs_path = tempfile.mkdtemp('graphs')
        writer = tf.summary.FileWriter(graphs_path, tf.Session().graph)
        writer.close()
        import subprocess
        subprocess.call(['tensorboard', '--logdir', graphs_path])
        return

    tensor_output, tensor_input = load_graph(pb_path, tensor_output_name, tensor_input_name)

    if os.path.isdir(dataset_loader):
        loader_dir = dataset_loader
    else:
        loader_dir = os.path.dirname(dataset_loader)

    sys.path.append(os.path.abspath(loader_dir))
    loader = __import__('loader')
    dataset_val = loader.load_dataset(args)

    dataset = {dataset_input_name: dataset_val}
    dataset = overwride_is_training(dataset)

    (output_code, output_bin) = convert(
        tensor_output, tensor_input,
        dataset,
        eight_bit_mode=eight_bit_mode,
        input_minmax_auto=input_minmax_auto,
        input_min=input_min,
        input_max=input_max,
        prefix=prefix,
        layer_start_idx=layer_start_idx
    )
    c_file, h_file = output_code

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path + '.c', 'w') as of:
        of.write(c_file)
    print('generate {} finish'.format(output_path + '.c'))

    with open(output_path + '.h', 'w') as of:
        of.write(h_file)
    print('generate {} finish'.format(output_path + '.h'))

    if output_bin is not None:
        os.makedirs(os.path.dirname(output_bin_name), exist_ok=True)
        with open(output_bin_name, 'wb') as of:
            of.write(output_bin)
        print('generate bin finish')


if __name__ == '__main__':
    main()
