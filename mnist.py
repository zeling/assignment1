import autodiff as ad
import numpy as np

def softmax(x):
    exps = ad.exp_op(x)
    return ad.mul_op(exps, ad.reciprocal_op(ad.sum_op(exps)))

if __name__ ==  "__main__":
    x = ad.Variable("x")
    exps = ad.exp_op(x)
    sum_of_exps = ad.sum_op(exps)
    reciprocal_of_sum = ad.reciprocal_op(sum_of_exps)
    probs = ad.mul_op(exps, reciprocal_of_sum)
    grad_x, grad_exps, grad_sum, grad_reci, = ad.gradients(probs, [x, exps, sum_of_exps, reciprocal_of_sum])
    executor = ad.Executor([probs, x, grad_x, exps, grad_exps, sum_of_exps, grad_sum, reciprocal_of_sum, grad_reci])
    print(executor.run({x : np.array([1., 2., 5.])}))
