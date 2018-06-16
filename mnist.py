import autodiff as ad
import numpy as np

def softmax(x):
    x_max = ad.max_op(x)
    x_shift = ad.add_byscalar_op(ad.neg_op(x_max), x)
    exps = ad.exp_op(x_shift)
    return ad.mul_byscalar_op(ad.reciprocal_op(ad.sum_op(exps)), exps)

def minus_op(x, y):
    return ad.add_op(x, ad.neg_op(y))

def softmax_ce_loss(preds, truth):
    """
    calculate the softmax and xent loss in a more efficient way
    :param preds: the pred is the output of the model 
    :param truth: the true label, a one-hot vector
    :return: the loss
    """
    pred_max = ad.max_op(preds)
    preds_shift = ad.add_byscalar_op(ad.neg_op(pred_max), x)
    exps = ad.exp_op(preds_shift)
    return minus_op(ad.log_op(ad.sum_op(exps)), ad.sum_op(ad.mul_op(preds_shift, truth)))

if __name__ ==  "__main__":
    x = ad.Variable("x")
    # y = ad.Variable("y")
    # z = ad.mul_byscalar_op(ad.max_op(x), y)
    y = softmax(x)
    label = ad.Variable("label")
    z = softmax_ce_loss(x, label)
    grad_x, = ad.gradients(z, [x])
    executor = ad.Executor([y, z, grad_x])
    print(executor.run({x : np.array([1., 2., 5.]), label: np.array([0, 0, 1])}))

    from mxnet import nd, autograd
    a = nd.array([[1., 2., 5.]])
    label = nd.array([2])
    a.attach_grad()
    with autograd.record():
        l = nd.softmax_cross_entropy(a, label)
        l.backward()
    print(l, a.grad)
