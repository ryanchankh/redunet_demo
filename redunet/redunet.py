import torch
import torch.nn as nn
from loss import compute_mcr2
from .layers.redulayer import ReduLayer


class ReduNet(nn.Sequential):
    def __init__(self, *modules):
        super(ReduNet, self).__init__(*modules)


    def init(self, inputs, labels, batch_size=0):
        with torch.no_grad():
            return self.forward(inputs,
                                labels,
                                batch_size,
                                init=True,
                                loss=True)

    def update(self, inputs, labels, tau=0.1, batch_size=0):
        with torch.no_grad():
            return self.forward(inputs, 
                                labels, 
                                batch_size, 
                                tau=tau,
                                update=True)

    def zero(self):
        with torch.no_grad():
            for module in self:
                if isinstance(module, ReduLayer):
                    module.zero()

    def forward(self,
                inputs, 
                labels=None,
                batch_size=0,
                tau=0.1, 
                init=False,
                update=False,
                loss=True):
        preprocessed, postprocessed = False, False
        self.losses = {'layer': [], 'loss_total':[], 'loss_expd': [], 'loss_comp': []}

        for layer_i, module in enumerate(self):
            # preprocess for redunet layers
            if not preprocessed and self._isReduLayer(module):
                inputs = module.preprocess(inputs)
                preprocessed = True

            # init model
            if init and self._isReduLayer(module):
                if batch_size > 0:
                    idx = torch.arange(layer_i*batch_size, (layer_i+1)*batch_size)
                    batch_inputs, batch_labels = inputs[idx], labels[idx]
                    module.init(batch_inputs, batch_labels)
                else:
                    module.init(inputs, labels)

            # update model
            if update and self._isReduLayer(module):
                if batch_size > 0:
                    idx = torch.arange(layer_i*batch_size, (layer_i+1)*batch_size)
                    batch_inputs, batch_labels = inputs[idx], labels[idx]
                    module.update(batch_inputs, batch_labels, tau)
                else:
                    module.update(inputs, labels, tau)

            # forward
            if self._isReduLayer(module):
                inputs, preds = module(inputs, return_y=True)
            else:
                inputs = module(inputs)

            # compute loss for redunet layer
            if loss and self._isReduLayer(module):
                losses = compute_mcr2(inputs, preds, module.eps)
                self.append_loss(layer_i, *losses)

            # postprocess for redunet layers
            if preprocessed and not postprocessed:
                if layer_i == len(self) - 1 and self._isReduLayer(module): # last layer
                    inputs = module.postprocess(inputs)
                    preprocessed, postprocessed = False, True
                elif not self._isReduLayer(self[layer_i+1]): # not last layer but next layer is not redunet instance
                    inputs = module.postprocess(inputs)
                    preprocessed, postprocessed = False, True

        return inputs


    def get_loss(self):
        return self.losses

    def append_loss(self, layer_i, loss_total, loss_expd, loss_comp):
        self.losses['layer'].append(layer_i)
        self.losses['loss_total'].append(loss_total)
        self.losses['loss_expd'].append(loss_expd)
        self.losses['loss_comp'].append(loss_comp)
        # print(f"{layer_i} | {loss_total:.6f} {loss_expd:.6f} {loss_comp:.6f}")

    def _isReduLayer(self, module):
        return isinstance(module, ReduLayer)
