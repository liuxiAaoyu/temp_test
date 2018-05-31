import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity


def _make_dense_layer(growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential(prefix='')
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(bn_size*growth_rate,
                               kernel_size=1, use_bias=False))
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(growth_rate, kernel_size=3,
                               padding=1, use_bias=False))
    if dropout:
        new_features.add(nn.Dropout(dropout))
    out = HybridConcurrent(axis=1, prefix='')
    out.add(Identity())
    out.add(new_features)
    return out


def _make_dense_layer_x(dilate, growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential(prefix='')
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(bn_size*growth_rate,
                               kernel_size=1, use_bias=False))
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(growth_rate, kernel_size=3,
                               padding=dilate, dilation=dilate, use_bias=False))
    if dropout:
        new_features.add(nn.Dropout(dropout))
    out = HybridConcurrent(axis=1, prefix='')
    out.add(Identity())
    out.add(new_features)
    return out


def _make_dense_block(num_layers, bn_size, growth_rate, dropout, stage_index):
    out = nn.HybridSequential(prefix='stage%d_' % stage_index)
    with out.name_scope():
        for _ in range(num_layers):
            out.add(_make_dense_layer(growth_rate, bn_size, dropout))
    return out


def _make_dense_block_x(dalites, num_layers, bn_size, growth_rate, dropout, stage_index):
    out = nn.HybridSequential(prefix='stage%d_' % stage_index)
    with out.name_scope():
        for i in range(num_layers):
            out.add(_make_dense_layer_x(
                dalites[i], growth_rate, bn_size, dropout))
    return out


def _make_transition(num_output_features):
    out = nn.HybridSequential(prefix='')
    out.add(nn.BatchNorm())
    out.add(nn.Activation('relu'))
    out.add(nn.Conv2D(num_output_features, kernel_size=1, use_bias=False))
    out.add(nn.AvgPool2D(pool_size=2, strides=2))
    return out


def _make_transition_o(num_output_features):
    out = nn.HybridSequential(prefix='')
    out.add(nn.BatchNorm())
    out.add(nn.Activation('relu'))
    out.add(nn.Conv2D(num_output_features, kernel_size=1, use_bias=False))
    return out


class DUC(HybridBlock):
    def __init__(self, label_num, ignore_lable, aspp_num, aspp_stride, cell_cap, **kwargs):
        super(DUC, self).__init__(**kwargs)
        self.label_num = label_num
        self.aspp_num = aspp_num
        # for i in range(aspp_num):
        #     pad = ((i+1)*aspp_stride, (i+1)*aspp_stride)
        #     dilate = pad
        #     conv_aspp = nn.Conv2D(cell_cap*label_num, kernel_size=3,
        #                           padding=dilate, dilation=dilate,)
        #     self.aspp_list.append(conv_aspp)
        pad = ((1)*aspp_stride, (1)*aspp_stride)
        dilate = pad
        self.conv1 = nn.Conv2D(cell_cap*label_num, kernel_size=3,
                               padding=dilate, dilation=dilate)
        pad = ((2)*aspp_stride, (2)*aspp_stride)
        dilate = pad
        self.conv2 = nn.Conv2D(cell_cap*label_num, kernel_size=3,
                               padding=dilate, dilation=dilate)
        pad = ((3)*aspp_stride, (3)*aspp_stride)
        dilate = pad
        self.conv3 = nn.Conv2D(cell_cap*label_num, kernel_size=3,
                               padding=dilate, dilation=dilate)
        pad = ((4)*aspp_stride, (4)*aspp_stride)
        dilate = pad
        self.conv4 = nn.Conv2D(cell_cap*label_num, kernel_size=3,
                               padding=dilate, dilation=dilate)

    def hybrid_forward(self, F, x):
        summ = F.ElementWiseSum(
            *[self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)])
        cls_score_reshape = F.reshape(data=summ, shape=(
            0, self.label_num, -1), name='cls_score_reshape')
        return cls_score_reshape


densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}


class DenseNet_x(HybridBlock):
    def __init__(self, bn_size=4, dropout=0, classes=2, **kwargs):
        super(DenseNet_x, self).__init__(**kwargs)
        with self.name_scope():
            num_init_features, growth_rate, block_config = densenet_spec[121]
            self.features = nn.HybridSequential(prefix='')
            # self.features.add(nn.Conv2D(
            #     num_init_features, kernel_size=7, strides=2, padding=3, use_bias=False))
            # self.features.add(nn.BatchNorm())
            # self.features.add(nn.Activation('relu'))
            # self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            num_features = num_init_features
            # self.features.add(_make_dense_block(
            #     block_config[0], bn_size, growth_rate, dropout, 1))
            num_features = num_features + block_config[0]*growth_rate
            # self.features.add(_make_transition(num_features//2))

            num_features = num_features//2
            # self.features.add(_make_dense_block(
            #     block_config[1], bn_size, growth_rate, dropout, 2))

            pre_features = mx.gluon.model_zoo.vision.densenet121(
                pretrained=True).features
            for i in range(7):
                self.features.add(pre_features[i])

            self.features_x = nn.HybridSequential(prefix='')

            num_features = num_features + block_config[1]*growth_rate
            self.features_x.add(_make_transition_o(num_features//2))
            num_features = num_features//2

            dalites = [1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5,
                       9, 1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5, 9]
            self.features_x.add(_make_dense_block_x(
                dalites, block_config[2], bn_size, growth_rate, dropout, 3))
            num_features = num_features + block_config[2]*growth_rate
            self.features_x.add(_make_transition_o(num_features//2))
            num_features = num_features//2

            dalites = [2, 5, 9, 17, 2, 5, 9, 17, 2, 5, 9, 17, 2, 5, 9, 17]
            self.features_x.add(_make_dense_block_x(
                dalites, block_config[3], bn_size, growth_rate, dropout, 4))

            self.features_x.add(self._make_layer(DUC, 'duc'))

    def _make_layer(self, block, stage_index, label_num=2, ignore_lable=255, aspp_num=4, aspp_stride=6, cell_cap=64):
        layer = nn.HybridSequential(prefix='stage%s_' % stage_index)
        with layer.name_scope():
            layer.add(block(label_num, ignore_lable,
                            aspp_num, aspp_stride, cell_cap,))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.features_x(x)
        return x


if __name__ == '__main__':
    net = DenseNet_x()
    #net.features.initialize(mx.init.Xavier())
    net.features_x.initialize(mx.init.Xavier())
    net.hybridize()
    x = mx.nd.random_normal(shape=(16, 3, 480, 480))
    for i in net.features:
        x = i(x)
        print(i.name, x.shape)
    for i in net.features_x:
        x = i(x)
        print(i.name, x.shape)

    x = mx.nd.random_normal(shape=(16, 3, 480, 480))
    pre_net = mx.gluon.model_zoo.vision.densenet121(pretrained=True)
    for i in pre_net.features:
        x = i(x)
        print(i.name, x.shape)
    #print(net.features[0].weight.data())
    x = mx.nd.zeros(shape=(16, 3, 480, 480))
    net(x)

    # names = [i for i in net.features_x[0].collect_params()]
    # for i in names:
    #     j=i[len(net.features_x.prefix):]
    #     j=j[:j.rfind('0')]+j[j.rfind('0'):].replace('0','3')
    #     #net.features_x.params.get(i[len(net.features_x.prefix):]).initialize()
    #     net.features_x.params.get(i[len(net.features_x.prefix):]).set_data(pre_net.features.params.get(j))
    
    # names = [i for i in net.features_x[1].collect_params()]
    # for i in names:
    #     net.features_x.params.get(i).set_data(pre_net.features.params.get(pre_net.features.prefix+i[len(net.features_x.prefix):]))
    
    # names = [i for i in net.features_x[2].collect_params()]
    # for i in names:
    #     j=pre_net.features.prefix+i[len(net.features_x.prefix):]
    #     j=j[:j.rfind('1')]+j[j.rfind('1'):].replace('1','4')
    #     net.features_x.params.get(i).set_data(pre_net.features.params.get(j))
    
    # names = [i for i in net.features_x[3].collect_params()]
    # for i in names:
    #     net.features_x.params.get(i).set_data(pre_net.features.params.get(pre_net.features.prefix+i[len(net.features_x.prefix):]))
    
    names1 = [i for i in net.collect_params()]
    names2 = [i for i in pre_net.collect_params()]

    net.save_params('densenet121HDCUDC.params')

    print(len(names1),len(names2))
    print("finish")
