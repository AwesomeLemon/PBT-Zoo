import torch


def get_optimizer(cfg, model, solution_optimizer_vals):
    if cfg.task.optimizer.name == 'SGD':
        kwargs = {'lr': cfg.task.optimizer.lr,
                  'momentum': cfg.task.optimizer.momentum,
                  'weight_decay': cfg.task.optimizer.weight_decay}
        kwargs.update(solution_optimizer_vals)
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif cfg.task.optimizer.name == 'Adam':
        kwargs = {'lr': cfg.task.optimizer.lr,
                  'weight_decay': cfg.task.optimizer.weight_decay}
        kwargs.update(solution_optimizer_vals)
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif cfg.task.optimizer.name == 'AdamW':
        kwargs = {'lr': cfg.task.optimizer.lr,
                  'weight_decay': cfg.task.optimizer.weight_decay}
        kwargs.update(solution_optimizer_vals)
        return torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise NotImplementedError