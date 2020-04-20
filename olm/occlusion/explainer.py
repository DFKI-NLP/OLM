import numpy as np
from torch.autograd import Variable, Function
import torch

# explainers taken from: https://github.com/yulongwang12/visual-attribution/blob/master/explainer/backprop.py


class VanillaGradExplainer:
    def __init__(self, model, input_key, output_getter=None):
        self.model = model
        self.input_key = input_key
        self.output_getter = output_getter

    def _backprop(self, inp, ind):
        output = self.model(**inp)
        if self.output_getter is not None:
            output = self.output_getter(output)
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        self.model.zero_grad()
        output.backward(grad_out)
        return inp[self.input_key].grad.data

    def explain(self, inp, ind):
        return self._backprop(inp, ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model, input_key=None, output_getter=None):
        super().__init__(model=model,
                         input_key=input_key,
                         output_getter=output_getter)

    def explain(self, inp, ind):
        grad = self._backprop(inp, ind)
        return inp[self.input_key].data * grad


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model, input_key=None, output_getter=None):
        super().__init__(model=model,
                         input_key=input_key,
                         output_getter=output_getter)

    def explain(self, inp, ind):
        grad = self._backprop(inp, ind)
        return grad.abs()


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100, input_key=None, output_getter=None):
        super().__init__(model=model,
                         input_key=input_key,
                         output_getter=output_getter)
        self.steps = steps

    def explain(self, inp, ind):
        grad = 0
        inp_data = inp[self.input_key].data.clone()

        for alpha in np.linspace(0, 1.0, num=self.steps, endpoint=False):
            new_inp = inp.copy()
            new_inp[self.input_key] = Variable(inp_data * alpha, requires_grad=True)
            g = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / self.steps
