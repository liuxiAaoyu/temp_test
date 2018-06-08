import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity


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


class BottleneckV2_x(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    `_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    depper : bool, default False
        Whether to depper the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, dilate, depper=False, in_channels=0, **kwargs):
        super(BottleneckV2_x, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm(use_global_stats=True)
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm(use_global_stats=True)
        #self.conv2 = _conv3x3(channels//4, stride, channels//4)
        self.conv2 = nn.Conv2D(channels//4, kernel_size=3, strides=1, padding=dilate, dilation=dilate,
                            use_bias=False, in_channels=channels//4)
        self.bn3 = nn.BatchNorm(use_global_stats=True)
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if depper:
            self.depper = nn.Conv2D(channels, 1, strides=1, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.depper = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.depper:
            residual = self.depper(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual

class ResNetV2_x(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    `_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block=BottleneckV2_x, layers=[3, 4, 23, 3], channels=[64, 256, 512, 1024, 2048], classes=19, thumbnail=False, **kwargs):
        super(ResNetV2_x, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            
            pre_features = mx.gluon.model_zoo.vision.resnet101_v2(
                pretrained=True).features
            for i in range(7):
                self.features.add(pre_features[i])
            
            self.features_x = nn.HybridSequential(prefix='')

            dalites = [1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5,
                       9, 1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5]
            self.features_x.add(self._make_layer(block, dalites, layers[2], channels[3],stage_index=3,in_channels=channels[2]))

            dalites = [5, 9, 17]
            self.features_x.add(self._make_layer(block, dalites, layers[3], channels[4],stage_index=4,in_channels=channels[3]))

            self.features_x.add(self._make_layer_DUC(DUC, 'duc', classes))

    def _make_layer(self, block, dalites, layers, channels, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, dilate=dalites[0], depper=True, in_channels=in_channels,
                            prefix=''))
            for i in range(layers-1):
                layer.add(block(channels, dilate=dalites[i+1], depper=False, in_channels=channels, prefix=''))
        return layer
        
    def _make_layer_DUC(self, block, stage_index, label_num=2, ignore_lable=255, aspp_num=4, aspp_stride=6, cell_cap=16):
        layer = nn.HybridSequential(prefix='stage%s_' % stage_index)
        with layer.name_scope():
            layer.add(block(label_num, ignore_lable,
                            aspp_num, aspp_stride, cell_cap))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.features_x(x)
        return x



if __name__ == '__main__':
    net = ResNetV2_x(classes=19)
    #net.features.initialize(mx.init.Xavier())
    net.features_x.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34))
    net.hybridize()
    x = mx.nd.random_normal(shape=(16, 3, 480, 480))
    for i in net.features:
        x = i(x)
        print(i.name, x.shape)
    for i in net.features_x:
        x = i(x)
        print(i.name, x.shape)

    x = mx.nd.random_normal(shape=(16, 3, 480, 480))
    pre_net = mx.gluon.model_zoo.vision.resnet101_v2(pretrained=True)
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
    mx.viz.plot_network(net(mx.sym.var("data")), shape={'data': (3, 3, 480, 480)}).view()
    net.save_params('resnet101HDCUDC.params')

    print(len(names1),len(names2))
    print("finish")
