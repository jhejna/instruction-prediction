import torch
from .reptile import Reptile


class FOMAML(Reptile):

    def __init__(self, *args, **kwargs):
        super(self, FOMAML).__init__(*args, **kwargs)
    
    def _adapt(self, support_set):
        # Adapt the fast network parameters based on the support set
        for i in range(self.inner_iters):
            last_backup = self.network.state_dict()
            current_batch = {k: v[i*self.batch_size:(i+1)*self.batch_size] for k, v in support_set.items()}
            self.optim.zero_grad()
            loss, metrics = self._compute_loss(current_batch, action_coeff=self.action_coeff, lang_coeff=self.lang_coeff)
            loss.backward()
            self.optim.step()
        return metrics, last_backup

    def _training_step(self, meta_batch, loss_lists):
        """Take a single training step update with First Order MAML.
        """
        updates = []
        for support_set, _ in meta_batch:
            # update the fast weights
            metrics, last_backup = self._adapt(support_set)
            # Record most recent metrics for the sample task.
            for metric_name, metric_value in metrics.items():
                loss_lists[metric_name].append(metrics[metric_name])
            update = {}
            current_backup = self.network.state_dict()
            for k in last_backup:
                assert k in current_backup, "self.network.state_dict() keys differ for the same network"
                update[k] = current_backup[k] - last_backup[k]
            updates.append(update)
            self.network.load_state_dict(self.meta_network.state_dict())

        if len(updates) == 1:
            updates = updates[0]
        else:
            #  We have a meta batch size larger than one and need to average the weights
            with torch.no_grad():
                avg_updates = {}
                for k in updates[0]:
                    avg_updates[k] = sum([update[k] for update in updates]) / len(updates)
            updates = avg_updates

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

