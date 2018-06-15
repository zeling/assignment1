import autodiff as ad
import numpy as np

def softmax(x):
    x_max = ad.max_op(x)
    x_shift = ad.add_byscalar_op(ad.neg_op(x_max), x)
    exps = ad.exp_op(x_shift)
    return ad.mul_byscalar_op(ad.reciprocal_op(ad.sum_op(exps)), exps)

if __name__ ==  "__main__":
    x = ad.Variable("x")
    # y = ad.Variable("y")
    # z = ad.mul_byscalar_op(ad.max_op(x), y)
    y = softmax(x)
    grad_x, = ad.gradients(y, [x])
    executor = ad.Executor([y, grad_x])
    print(executor.run({x : np.array([2., 2., 2.])}))
