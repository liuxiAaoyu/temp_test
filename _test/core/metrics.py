import mxnet as mx
import numpy as np

class AccWithIgnoreMetric(mx.metric.EvalMetric):
    def __init__(self, ignore_label, name='AccWithIgnore'):
        super(AccWithIgnoreMetric, self).__init__(name=name)
        self._ignore_label = ignore_label
    
    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
            mx.metric.check_label_shapes(label, pred_label)
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat) - (label.flat == self._ignore_label).sum()


class TPAccMetric(mx.metric.EvalMetric):
    def __init__(self, ignore_label, name='TPAcc'):
        super(TPAccMetric, self).__init__(name=name)
        self._ignore_label = ignore_label
    
    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
            mx.metric.check_label_shapes(label, pred_label)

            pred_cur = (pred_label.flat == 1)
            gt_cur = (label.flat == 1)

            tp = np.logical_and(pred_cur, gt_cur).sum()
            gtp = gt_cur.sum()
            self.sum_metric += tp
            self.num_inst += gtp+0.001


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
    # def update(self, labels, preds):
    #     mx.metric.check_label_shapes(labels, preds)
    #     for i in range(len(labels)):
    #         pred_label = mx.ndarray.argmax_channel(preds[i])
    #         label = labels[i]
    #         pred_label=mx.ndarray.cast(pred_label, dtype='int32')
    #         label=mx.ndarray.cast(label, dtype='int32')
    #         #mx.metric.check_label_shapes(label, pred_label)

    #         iou = 0
    #         eps = 1e-6
    #         # skip_label_num = 0
    #         for j in range(self._label_num):
    #             pred_cur = (pred_label.reshape((-1)) == j)
    #             gt_cur = (label.reshape((-1)) == j)
    #             tp = mx.ndarray.logical_and(pred_cur, gt_cur).sum()
    #             denom = mx.ndarray.logical_or(pred_cur, gt_cur).sum() - mx.ndarray.logical_and(pred_cur, label.reshape((-1)) == self._ignore_label).sum()
    #             print(tp,denom)
    #             assert tp <= denom
    #             self._tp[j] += tp
    #             self._denom[j] += denom
    #             iou += self._tp[j] / (self._denom[j] + eps)
    #         iou /= self._label_num
    #         self.sum_metric = iou
    #         self.num_inst = 1


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
            predsi=mx.nd.softmax(preds[i],axis=1)
            prediction = predsi.asnumpy()[:]
            shape = prediction.shape
            if len(shape) == 4:
                shape = (shape[0], shape[1], shape[2]*shape[3])
                prediction = prediction.reshape(shape)
            label = labels[i].asnumpy()
            soft_label = np.zeros(prediction.shape)

            for b in range(soft_label.shape[0]):
                for c in range(self._label_num):
                    soft_label[b][c][label[b][0] == c] = 1.0
            #a=np.ones_like(prediction[soft_label == 1])
            #a= a*eps
            #b=a<prediction[soft_label == 1]
            #c=prediction[soft_label == 1]*b           
            #b=a>=prediction[soft_label == 1]
            #loss += (-np.log(c + eps*b)).sum()
            loss += (-np.log(prediction[soft_label == 1] + eps)).sum()
            cnt += prediction[soft_label == 1].size
        self.sum_metric += loss
        self.num_inst += cnt 
