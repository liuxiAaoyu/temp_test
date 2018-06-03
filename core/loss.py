from mxnet import ndarray
from mxnet import gluon


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


class MySoftmaxCrossEntropyLoss(gluon.loss.Loss):
    r"""Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

    If `sparse_label` is `True` (default), label should contain integer
    category indicators:

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        p = \softmax({pred})

        L = -\sum_i \log p_{i,{label}_i}

    `label`'s shape should be `pred`'s shape with the `axis` dimension removed.
    i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape should
    be (1,2,4).

    If `sparse_label` is `False`, `label` should contain probability distribution
    and `label`'s shape should be the same with `pred`:

    .. math::

        p = \softmax({pred})

        L = -\sum_i \sum_j {label}_j \log p_{ij}

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, label_calsses = 2, **kwargs):
        super(MySoftmaxCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._label_calsses = label_calsses

    def hybrid_forward(self, F, pred, label):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)

        labels=[]
        for i in range(self._label_calsses):
            labels.append( F.sum(label == i) )
        labels_sum = F.add_n(*labels)
        label_weights=[]
        for i in range(self._label_calsses):
            label_num = F.sum(label == i)
            label_weights.append( F.broadcast_mul(
                (label == i), 1/F.log(label_num/(labels_sum+0.0000001)+1.02)) )
        sample_weight = F.add_n(*label_weights)

        loss = _apply_weighting(
            F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


