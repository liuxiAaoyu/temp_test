import mxnet as mx

class AccWithIgnoreMetric(mx.metric.EvalMetric):
    def __init__(self, ignore_lable, name='AccWithIgnore'):
        super(AccWithIgnoreMetric, self).__init__(name=name)
        self._ignore_label = ignore_lable
    
    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channels(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat) - (label.flat == self._ignore_label).sum()


class IoUMetric(mx.metric.EvalMetric):
    def __init__(self, ignore_label, label_num, name='IoU'):
        self._ignore_label = ignore_label
        self._label_num = label_num
        super(IoUMetric, self).__init__(name=name)

    def reset(self):
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            iou = 0
            eps = 1e-6
            # skip_label_num = 0
            for j in range(self._label_num):
                pred_cur = (pred_label.flat == j)
                gt_cur = (label.flat == j)
                tp = np.logical_and(pred_cur, gt_cur).sum()
                denom = np.logical_or(pred_cur, gt_cur).sum() - np.logical_and(pred_cur, label.flat == self._ignore_label).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self._label_num
            self.sum_metric = iou
            self.num_inst = 1


class SoftmaxLoss(mx.metric.EvalMetric):
    def __init__(self, ignore_label, label_num, name='OverallSoftmaxLoss'):
        super(SoftmaxLoss, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._label_num = label_num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        loss = 0.0
        cnt = 0.0
        eps = 1e-6
        for i in range(len(labels)):
            prediction = preds[i].asnumpy()[:]
            shape = prediction.shape
            if len(shape) == 4:
                shape = (shape[0], shape[1], shape[2]*shape[3])
                prediction = prediction.reshape(shape)
            label = labels[i].asnumpy()
            soft_label = np.zeros(prediction.shape)
            for b in range(soft_label.shape[0]):
                for c in range(self._label_num):
                    soft_label[b][c][label[b] == c] = 1.0

            loss += (-np.log(prediction[soft_label == 1] + eps)).sum()
            cnt += prediction[soft_label == 1].size
        self.sum_metric += loss
        self.num_inst += cnt 