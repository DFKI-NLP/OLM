from typing import List

import torch
import numpy as np
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.predictors import Predictor
from allennlp.nn import util
from torch.utils.hooks import RemovableHandle


class AllenNLPVanillaGradExplainer:
    def __init__(self, predictor: Predictor, output_getter=None):
        self.predictor = predictor
        self.output_getter = output_getter

    def _backprop(self, instances: List[Instance], ind: torch.Tensor, register_forward_hooks: bool = True):
        embedding_gradients: List[torch.Tensor] = []
        grad_hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        embedding_values: List[torch.Tensor] = []
        if register_forward_hooks:
            val_hooks: List[RemovableHandle] = self._register_embedding_value_hooks(embedding_values)

        model = self.predictor._model

        cuda_device = model._get_prediction_device()

        dataset = Batch(instances)
        dataset.index_instances(model.vocab)

        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)

        output = model.decode(
                model.forward(**model_input)  # type: ignore
        )

        if self.output_getter is not None:
            output = self.output_getter(output)

        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)

        model.zero_grad()
        output.backward(grad_out)

        for hook in grad_hooks:
            hook.remove()

        if register_forward_hooks:
            for hook in val_hooks:
                hook.remove()

        return embedding_values, embedding_gradients

    def explain(self, instances, ind):
        return self._backprop(instances, ind)[1]

    def _register_embedding_gradient_hooks(self, embedding_gradients):

        def backward_hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = util.find_embedding_layer(self.predictor._model)
        # workaround, otherwise forward hook is not called on embedding
        embedding_layer.weight.requires_grad = True
        backward_hooks.append(embedding_layer.register_backward_hook(backward_hook_layers))
        return backward_hooks

    def _register_embedding_value_hooks(self, embedding_values):

        def forward_hook_layers(module, input, output):
            embedding_values.append(output)

        forward_hooks = []
        embedding_layer = util.find_embedding_layer(self.predictor._model)
        # workaround, otherwise forward hook is not called on embedding
        embedding_layer.weight.requires_grad = True
        forward_hooks.append(embedding_layer.register_forward_hook(forward_hook_layers))
        return forward_hooks


class AllenNLPGradxInputExplainer(AllenNLPVanillaGradExplainer):
    def __init__(self, predictor, output_getter=None):
        super().__init__(predictor=predictor,
                         output_getter=output_getter)

    def explain(self, instances, ind):
        inputs, grads = self._backprop(instances, ind)
        return [input * grad for input, grad in zip(inputs, grads)]


class AllenNLPSaliencyExplainer(AllenNLPVanillaGradExplainer):
    def __init__(self, predictor, output_getter=None):
        super().__init__(predictor=predictor,
                         output_getter=output_getter)

    def explain(self, instances, ind):
        _, grads = self._backprop(instances, ind)
        return [grad.abs() for grad in grads]


class AllenNLPIntegrateGradExplainer(AllenNLPVanillaGradExplainer):
    def __init__(self, predictor, steps=100, output_getter=None):
        super().__init__(predictor=predictor,
                         output_getter=output_getter)
        self.steps = steps

    def explain(self, instances, ind):
        grads = [0 for _ in instances]
        inputs = []

        for alpha in np.linspace(0, 1.0, num=self.steps, endpoint=False):
            handle: RemovableHandle = self._register_embedding_value_hook(alpha, inputs)

            _, grads_current = self._backprop(instances, ind, register_forward_hooks=False)

            handle.remove()

            grads = [grad + grad_c for grad, grad_c in zip(grads, grads_current)]

        return [input * grad / self.steps for input, grad in zip(inputs, grads)]

    def _register_embedding_value_hook(self, alpha: float, embedding_values):

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embedding_values.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)

        embedding_layer = util.find_embedding_layer(self.predictor._model)
        embedding_layer.weight.requires_grad = True
        return embedding_layer.register_forward_hook(forward_hook)
