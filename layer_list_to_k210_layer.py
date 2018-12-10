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

import math

import k210_layer
import tensor_list_to_layer_list
import numpy as np
import tools


def make_k210_layer(iwo_minmax, ico_shapes, conv_weights_isdw, bn_mean_var_gamma_beta_epsilon, act_type,
                    pool_type_size_stride, eight_bit_mode=False, cbap_tensor_info=None, idx=-1):
    input_min, input_max, weights_min, weights_max, output_min, output_max = iwo_minmax
    input_shape, conv_shape, output_shape = ico_shapes
    conv_weights, conv_isdw = conv_weights_isdw
    conv_tensor_info, bn_tensor_info, act_tensor_info, pool_tensor_info, *_ = [
        *list(cbap_tensor_info or []), dict(), dict(), dict(), dict()
    ]

    output_name = pool_tensor_info.get('name') or act_tensor_info.get('name', 'noname')
    output_scale, output_bias = tools.min_max_to_scale_bias(output_min, output_max)
    layer_shape_trans = [
        int(input_shape[1]), int(input_shape[2]), int(input_shape[3]),
        int(output_shape[1]), int(output_shape[2]), int(output_shape[3])
    ]
    print(
        "[layer {}]".format(idx), output_name,
        'shape(WHC): {}x{}x{} => {}x{}x{}'.format(*layer_shape_trans),
        'scale/bias:', output_scale, output_bias
    )

    ret = k210_layer.K210Layer(eight_bit_mode)
    ret.conv = k210_layer.K210Conv(
        conv_weights,
        conv_isdw,
        eight_bit_mode, [input_shape, conv_shape],
        [input_min, input_max, weights_min, weights_max],
        tensor_info=conv_tensor_info
    )

    bn_mean, bn_var, bn_gamma, bn_beta, bn_epsilon = bn_mean_var_gamma_beta_epsilon
    ret.bn = k210_layer.K210BN(
        bn_mean,
        bn_var,
        bn_gamma,
        bn_beta,
        bn_epsilon,
        eight_bit_mode,
        tensor_info=bn_tensor_info
    )

    ret.act = k210_layer.K210Act(output_min, output_max, act_type,
                      eight_bit_mode=eight_bit_mode, tensor_info=act_tensor_info)

    if pool_type_size_stride is not None:
        pool_type, pool_size, pool_stride = pool_type_size_stride
        if pool_size == 2 and conv_shape[3] % 2 != 0:
            raise ValueError(
                "at {} unsupport padding mode SAME of pooling" \
                    .format(pool_tensor_info.get('name', 'noname'))
            )

        if conv_isdw and pool_size != 1:
            raise ValueError(
                'not supported DepthwiseConv2d({}) followed by pooling witch pool_size is not 1.' \
                    .format(pool_tensor_info.get('name', 'noname'))
            )

        ret.pool = k210_layer.K210Pool(pool_type, pool_size, pool_stride, pool_tensor_info)

    return ret


def make_k210_layer_from_tensor(sess, dataset, buffer, input_min, input_max, eight_bit_mode, range_from_batch, idx):
    pool_tensor_info = dict()
    pool_type_size_stride = None  # bypass pool

    if isinstance(buffer[-1], tensor_list_to_layer_list.LayerConvolutional) \
            or isinstance(buffer[-1], tensor_list_to_layer_list.LayerDepthwiseConvolutional):
        conv_layer = buffer.pop()

        input_shape = list(sess.run(conv_layer.tensor_conv_x, dataset).shape)
        conved_shape = list(sess.run(conv_layer.tensor_conv_y, dataset).shape)

        if conv_layer.tensor_conv_x.op.type == 'SpaceToBatchND':
            print('[warning] found SpaceToBatchND fix input_shape/')
            input_shape[1] = input_shape[1]-2
            input_shape[2] = input_shape[2]-2

        weights_min, weights_max, _ = range_from_batch(sess, conv_layer.tensor_conv_w, dataset, is_weights=True)
        conv_weights = conv_layer.weights
        conv_isdw = isinstance(conv_layer, tensor_list_to_layer_list.LayerDepthwiseConvolutional)
        conv_tensor_info = {'name': conv_layer.tensor_conv_y.name}

        if int(conv_layer.config['batch_normalize']) == 1:
            bn_mean_var_gamma_beta_epsilon = [
                conv_layer.batch_normalize_moving_mean,
                conv_layer.batch_normalize_moving_variance,
                conv_layer.batch_normalize_gamma,
                conv_layer.batch_normalize_beta,
                conv_layer.batch_normalize_epsilon
            ]
            bn_tensor_info = {'name': 'bn'}
        else:
            bias_shape = conv_layer.bias.shape
            bn_mean_var_gamma_beta_epsilon = [
                0, 1, np.ones(bias_shape), conv_layer.bias, 0
            ]
            bn_tensor_info = {'name': 'bn'}

        tensor_act = conv_layer.tensor_activation
        act_min_y, act_max_y, _ = range_from_batch(sess, tensor_act, dataset)
        act_type = conv_layer.config['activation']
        act_tensor_info = {'name': tensor_act.name if tensor_act is not None else 'default_linear'}
        output_shape = tensor_act.shape
    else:
        raise ValueError('unsupported type seq: ', *[type(l) for l in buffer])

    if len(buffer) > 0 and isinstance(buffer[-1], tensor_list_to_layer_list.LayerPool):
        pool_layer = buffer.pop()
        assert (isinstance(pool_layer, tensor_list_to_layer_list.LayerPool))
        pool_size = pool_layer.config['size']
        pool_stride = pool_layer.config['stride']
        pool_type = pool_layer.tensor_pool.op.type

        if pool_size == 2 and pool_layer.tensor_pool.op.inputs[0].shape[3] % 2 != 0:
            if pool_layer.tensor_pool.op.get_attr('padding') == b'SAME':
                raise ValueError("at {} unsupport padding mode SAME of pooling with size == 2" \
                                 .format(pool_layer.tensor_pool.name))

        pool_type_size_stride = [pool_type, pool_size, pool_stride]
        pool_tensor_info = {'name': pool_layer.tensor_pool.op.name}

    return {
        'iwo_minmax':[input_min, input_max, weights_min, weights_max, act_min_y, act_max_y],
        'ico_shapes':[input_shape, conved_shape, output_shape],
        'conv_weights_isdw':[conv_weights, conv_isdw],
        'bn_mean_var_gamma_beta_epsilon':bn_mean_var_gamma_beta_epsilon,
        'act_type':act_type,
        'pool_type_size_stride':pool_type_size_stride,
        'eight_bit_mode':eight_bit_mode,
        'cbap_tensor_info':[conv_tensor_info, bn_tensor_info, act_tensor_info, pool_tensor_info],
        'idx':idx
    }


def k210_layer_post_fix(kl_args_list: [k210_layer.K210Layer]):
    def fix_dw_with_strde2(kl_args_list: [k210_layer.K210Layer]):
        def expand_wh(shape_):
            shape_1 = shape_[1] * 2
            shape_2 = shape_[2] * 2
            return [shape_[0], shape_1, shape_2, shape_[3]]

        ret = []
        lack_of_left_pooling = False
        for kl_args in kl_args_list:
            input_shape, conv_shape, output_shape = kl_args['ico_shapes']
            conv_weights, conv_isdw = kl_args['conv_weights_isdw']
            pool_type_size_stride = kl_args['pool_type_size_stride']
            kl_args_fixed = dict(kl_args)

            conv_kernel_size = int(conv_weights.shape[0])
            conv_stride = int((int(input_shape[2])+1)/int(conv_shape[2]))

            if lack_of_left_pooling:
                if not conv_isdw and conv_kernel_size==1 and pool_type_size_stride is None:
                    # fix in current layer
                    input_shape = expand_wh(input_shape)
                    conv_shape = expand_wh(conv_shape)
                    lack_of_left_pooling = False
                    kl_args_fixed['pool_type_size_stride'] = ['leftPool', 2, 2]
                    kl_args_fixed['ico_shapes'] = [input_shape, conv_shape, output_shape]
                else:
                    if not (conv_kernel_size==1 and pool_type_size_stride is None):
                        raise ValueError(
                            'run fix_dw_with_strde2 failed. '+
                            'can not delay left_pooling over current layer, '+
                            'current layer conv_kernel_size:{}, pool_type_size_stride:{}'\
                            .format(conv_kernel_size, pool_type_size_stride)
                        )

                    # delay fix in after layers
                    input_shape = expand_wh(input_shape)
                    conv_shape = expand_wh(conv_shape)
                    output_shape = expand_wh(output_shape)
                    kl_args_fixed['ico_shapes'] = [input_shape, conv_shape, output_shape]


            if conv_stride == 2:
                if not conv_isdw:
                    if pool_type_size_stride is None:
                        # fix in current layer
                        conv_shape = expand_wh(conv_shape)
                        kl_args_fixed['pool_type_size_stride'] = ['leftPool', 2, 2]
                        kl_args_fixed['ico_shapes'] = [input_shape, conv_shape, output_shape]
                    else:
                        # fix later
                        lack_of_left_pooling = True
                        conv_shape = expand_wh(conv_shape)
                        output_shape = expand_wh(output_shape)
                        kl_args_fixed['ico_shapes'] = [input_shape, conv_shape, output_shape]
                else:
                    # dw layer needs to fix it later
                    lack_of_left_pooling = True
                    conv_shape = expand_wh(conv_shape)
                    output_shape = expand_wh(output_shape)
                    kl_args_fixed['ico_shapes'] = [input_shape, conv_shape, output_shape]


            ret.append(kl_args_fixed)

        if lack_of_left_pooling:
            raise ValueError('run fix_dw_with_strde2 failed. no more layers for fix.')
        return ret

    def fix_wh_leas_than_4(kl_args_list: [k210_layer.K210Layer]):
        def force_pad_to_4(shape_):
                return [shape_[0], 4, 4, shape_[3]]

        ret = []
        for kl_args in kl_args_list:
            input_shape, conv_shape, output_shape = kl_args['ico_shapes']
            kl_args_fixed = dict(kl_args)


            if input_shape[1] < 4 or conv_shape[1] < 4 or output_shape[1] < 4:
                input_shape = force_pad_to_4(input_shape)
                conv_shape = force_pad_to_4(conv_shape)
                output_shape = force_pad_to_4(output_shape)
                kl_args_fixed['ico_shapes'] = [input_shape, conv_shape, output_shape]

            ret.append(kl_args_fixed)

        return ret


    kl_args_list = fix_wh_leas_than_4(kl_args_list)
    kl_args_list = fix_dw_with_strde2(kl_args_list)
    return kl_args_list


def gen_k210_layers(layers: [tensor_list_to_layer_list.LayerBase], sess, dataset, range_from_batch,
                    eight_bit_mode=False, input_min=0, input_max=1, layer_start_idx=0):
    buffer = list(layers)
    buffer.reverse()
    kl_args_list = []

    net = buffer.pop()
    assert (isinstance(net, tensor_list_to_layer_list.LayerNet))

    while len(buffer) != 0:
        if len(kl_args_list) > 0:
            last_min = kl_args_list[-1]['iwo_minmax'][4]
            last_max = kl_args_list[-1]['iwo_minmax'][5]
        else:
            last_min = input_min
            last_max = input_max

        cur_k210_arg = make_k210_layer_from_tensor(
            sess=sess, dataset=dataset,
            buffer=buffer,
            input_min=last_min, input_max=last_max,
            eight_bit_mode=eight_bit_mode,
            range_from_batch=range_from_batch,
            idx=len(kl_args_list)+layer_start_idx
        )
        kl_args_list.append(cur_k210_arg)

    kl_args_fixed = k210_layer_post_fix(kl_args_list)
    kl_list = [make_k210_layer(**kl_args) for kl_args in kl_args_fixed]
    return kl_list
