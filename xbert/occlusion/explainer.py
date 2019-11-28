import numpy as np
from torch.autograd import Variable, Function
import torch

# explainers taken from: https://github.com/yulongwang12/visual-attribution/blob/master/explainer/backprop.py

class VanillaGradExplainer:
    def __init__(self, model, output_getter=None):
        self.model = model
        self.output_getter = output_getter

    def _backprop(self, inp, ind):
        output = self.model(inputs_embeds=inp)
        if self.output_getter is not None:
            output = self.output_getter(output)
        if ind is None:
            ind = output.data.max(1)[1]
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        output.backward(grad_out)
        return inp.grad.data

    def explain(self, inp, ind=None):
        return self._backprop(inp, ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model, output_getter=None):
        super().__init__(model=model,
                         output_getter=output_getter)

    def explain(self, inp, ind=None):
        grad = self._backprop(inp, ind)
        return inp.data * grad
    
    
class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model, output_getter=None):
        super().__init__(model=model,
                         output_getter=output_getter)

    def explain(self, inp, ind=None):
        grad = self._backprop(inp, ind)
        return grad.abs()


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100, output_getter=None):
        super().__init__(model=model,
                         output_getter=output_getter)
        self.steps = steps

    def explain(self, inp, ind=None):
        grad = 0
        inp_data = inp.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / self.steps
