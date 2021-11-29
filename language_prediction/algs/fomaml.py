from typing import OrderedDict
import torch
from .reptile import Reptile
from copy import deepcopy


class FOMAML(Reptile):

    def __init__(self, *args, **kwargs):
        super(FOMAML, self).__init__(*args, **kwargs)
    
    def _adapt(self, support_set, action_coeff=1.0, lang_coeff=0.0, is_eval=False):
        # Adapt the fast network parameters based on the support set
        if self.random_coeff and not is_eval:
            action_coeff, lang_coeff = sample_loss_coeffs()
            action_coeff *= self.action_coeff
            lang_coeff *= self.lang_coeff
        
        # Adapt the fast network parameters based on the support set
        for i in range(self.inner_iters):
            last_backup = deepcopy(self.network.state_dict())
            current_batch = {k: v[i*self.batch_size:(i+1)*self.batch_size] for k, v in support_set.items()}
            self.optim.zero_grad()
            loss, metrics = self._compute_loss(current_batch, action_coeff=self.action_coeff, lang_coeff=self.lang_coeff)
            loss.backward()
            self.optim.step()
        return metrics, last_backup

    def _training_step(self, meta_batch, loss_lists):
        """Take a single training step update with First Order MAML.
        """
        # Take a single first-order MAML training step
        updates = []
        for support_set, _ in meta_batch:
            # update the fast weights
            metrics, last_backup = self._adapt(support_set, action_coeff=self.action_coeff, lang_coeff=self.lang_coeff, is_eval=False)
            # Record most recent metrics for the sample task.
            for metric_name, metric_value in metrics.items():
                loss_lists[metric_name].append(metrics[metric_name])
            
            # Store query set weight delta
            update = deepcopy(self.network.state_dict())
            for k in update:
                update[k] -= last_backup[k] # Note that this is not the gradient, the gradient is Meta - Current
            updates.append(update)
            # Reset network weights for next step
            self.network.load_state_dict(self.meta_network.state_dict())

        if len(updates) == 1:
            updates = updates[0]
        else:
            avg_updates = deepcopy(self.network.state_dict())
            for k in avg_updates:
                avg_tensor = torch.zeros_like(avg_updates[k])
                for update in updates:
                    avg_tensor += update[k]
                avg_updates[k] = (avg_tensor / len(updates)).clone()
            updates = avg_updates
        # Need to invert the gradients in the updates
        for k in updates:
            updates[k] = -1*updates[k]
            
        # Hacky workaround to storing updates in self.network.parameters() for below gradient computation
        self.network.load_state_dict(updates)

        # Now set the gradients of the meta optimizer to be the weight differences
        self.meta_optim.zero_grad()
        for meta_param, param in zip(self.meta_network.parameters(), self.network.parameters()):
            if meta_param.grad is None:
                meta_param.grad = torch.autograd.Variable(torch.zeros(meta_param.size())).to(self.device)
            meta_param.grad.data.zero_() # perhaps unnecesary
            meta_param.grad.data.add_(param.data)
        
        self.meta_optim.step()
        # Now sync the weights of the two networks.
        self.network.load_state_dict(self.meta_network.state_dict())
