import numpy as np
import torch
from matplotlib import pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm

from dataset import get_dataset
from dataset.utils import get_loader
from optimizer import get_optimizer
import torch.utils.tensorboard as tb

from search_space.cs import ConfigSpaceSearchSpace
from utils.util_fns import adjust_optimizer_settings, convert_config_from_logarithmic, optimizer_to
from hydra.utils import instantiate
from torch.cuda.amp import autocast, GradScaler

class ClassificationTask:
    def __init__(self, cfg, search_space, **__):
        self.cfg = cfg
        self.search_space: ConfigSpaceSearchSpace = search_space
        self.dataset_wrapper = get_dataset(self.cfg)
        self.loss = torch.nn.CrossEntropyLoss()
        self.metric = Accuracy('multiclass', num_classes=self.cfg.task.data.n_classes)
        self.t_eval = self.cfg.task.t_eval # how often to eval for the curves
        self.viz = self.cfg.task.get('viz', False)


    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tensorboard_dir, only_evaluate):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric.reset()
        self.metric.to(device)
        if only_evaluate is None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = tb.SummaryWriter(tensorboard_dir)

        # create model
        model = instantiate(self.cfg.task.model)
        model.load_state_dict(cpkt_loaded['model_state_dict'])
        model.to(device)
        # create optimizer
        solution_optimizer_vals = self.get_optimizer_vals(solution)
        optimizer = get_optimizer(self.cfg, model, solution_optimizer_vals)
        optimizer.load_state_dict(cpkt_loaded['optimizer_state_dict']) # this will overwrite solution_optimizer_vals
        optimizer = adjust_optimizer_settings(optimizer, solution_optimizer_vals)
        optimizer_to(optimizer, device)

        # create transform
        dataset = self.dataset_wrapper.dataset
        transform_vals = self.get_transform_vals(solution)
        transforms_train = self.dataset_wrapper.create_train_transform(**transform_vals)
        if hasattr(dataset['train'], 'dataset'):
            dataset['train'].dataset.transform = transforms_train
        else:
            dataset['train'].transform = transforms_train

        # create loaders
        solution_loader_vals = self.get_loader_vals(solution)
        loaders = get_loader(self.cfg, dataset, seed, **solution_loader_vals)

        if only_evaluate is not None:
            assert type(only_evaluate) == list
            out = {}

            if 'val' in only_evaluate:
                val_acc, *_ = self._eval([], device, -1, loaders['val'], model, t, t_step, None)
                out['val'] = val_acc
                out['fitness'] = val_acc
            if 'test' in only_evaluate:
                test_acc, *_ = self._eval([], device, -1, loaders['test'], model, t, t_step, None)
                out['test'] = test_acc

            return out

        # scaler
        scaler = GradScaler()
        scaler.load_state_dict(cpkt_loaded['scaler_state_dict'])
        del cpkt_loaded

        iterator_train = iter(loaders['train'])

        losses = {'train': [], 'test': []}

        curve = []
        assert t_step % self.t_eval == 0
        for i_epoch in (pbar := tqdm(range(t_step // self.t_eval), desc=f'train+val')):
            # train
            model.train()
            for i_batch in range(self.t_eval):
                try:
                    img, lbl = next(iterator_train)
                except StopIteration:
                    iterator_train = iter(loaders['train'])
                    img, lbl = next(iterator_train)

                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()

                with autocast():
                    out = model(img)
                    loss = self.loss(out, lbl)
                losses['train'].append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if (i_batch + 1) % 10 == 0:
                    tb_writer.add_scalar('loss/train', np.mean(losses['train']), t + i_epoch * self.t_eval + i_batch)
                    losses['train'] = []

                if i_batch == 0 and i_epoch == 0 and self.viz:
                    n_viz = 16
                    img_viz_cur = img[:n_viz].detach().cpu()
                    mean, std = torch.Tensor(dataset['val'].mean), torch.Tensor(dataset['val'].std)
                    img_viz_cur = img_viz_cur * std[None, :, None, None] + mean[None, :, None, None]
                    train_image_viz = (img_viz_cur, lbl[:n_viz].detach().cpu(), out[:n_viz].detach().cpu())

            val_acc, val_image_viz, curve = self._eval(curve, device, i_epoch, loaders['val'], model, t, t_step, tb_writer)

        if self.viz:
            tb_writer.add_figure('images/val',
                              plot_classes_preds(*val_image_viz, self.dataset_wrapper.class_names),
                              global_step=t + t_step)

            tb_writer.add_figure('images/train',
                              plot_classes_preds(*train_image_viz, self.dataset_wrapper.class_names),
                              global_step=t + t_step)

        # test
        with torch.no_grad():
            for img, lbl in loaders['test']:
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                loss = self.loss(out, lbl)
                losses['test'].append(loss.item())
                self.metric(out, lbl)

        test_acc = self.metric.compute().item()
        self.metric.reset()
        tb_writer.add_scalar('acc/test', test_acc, t + t_step)
        tb_writer.add_scalar('loss/test', np.mean(losses['test']), t + t_step)
        losses['test'] = []

        optimizer_to(optimizer, 'cpu')
        dict_to_save = {'model_state_dict': model.cpu().state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict()}

        tb_writer.close()

        return {'fitness': curve[-1][1],
                'curve': curve,
                'dict_to_save': dict_to_save,
                'metrics': {'val': val_acc, 'test': test_acc}}

    def _eval(self, curve, device, i_epoch, loader_eval, model, t, t_step, tb_writer):
        model.eval()
        losses_eval = []
        val_image_viz = None

        with torch.no_grad():
            for i_batch, (img, lbl) in enumerate(loader_eval):
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                loss = self.loss(out, lbl)
                losses_eval.append(loss.item())
                self.metric(out, lbl)
                if i_batch == 0 and i_epoch == t_step // self.t_eval - 1 and self.viz:
                    n_viz = 16
                    img_viz_cur = img[:n_viz].detach().cpu()
                    mean, std = torch.Tensor(self.dataset_wrapper.mean), torch.Tensor(self.dataset_wrapper.std)
                    img_viz_cur = img_viz_cur * std[None, :, None, None] + mean[None, :, None, None]
                    val_image_viz = (img_viz_cur, lbl[:n_viz].detach().cpu(), out[:n_viz].detach().cpu())

        val_acc = self.metric.compute().item()
        self.metric.reset()
        if tb_writer is not None:
            tb_writer.add_scalar('acc/val', val_acc, t + i_epoch * self.t_eval)
            tb_writer.add_scalar('loss/val', np.mean(losses_eval), t + i_epoch * self.t_eval)
        curve.append((t + i_epoch * self.t_eval, val_acc))
        model.train()
        return val_acc, val_image_viz, curve

    def get_optimizer_vals(self, solution):
        config_dict = self.search_space.vector_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)
        config_dict = {k:v for k, v in config_dict.items() if k in ['lr', 'weight_decay', 'momentum']}
        return config_dict

    def get_loader_vals(self, solution):
        config_dict = self.search_space.vector_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)
        config_dict = {k:v for k, v in config_dict.items() if k in ['batch_size']}

        if 'batch_size' not in config_dict:
            config_dict['batch_size'] = self.cfg.task.data.batch_size

        return config_dict

    def get_transform_vals(self, solution):
        config_dict = self.search_space.vector_to_dict(solution)
        config_dict = convert_config_from_logarithmic(config_dict)
        config_dict = {k:v for k, v in config_dict.items() if k in ['rand_aug_n_ops', 'rand_aug_mag']}

        if 'rand_aug_n_ops' not in config_dict:
            config_dict['rand_aug_n_ops'] = self.cfg.task.data.rand_aug_n_ops

        if 'rand_aug_mag' not in config_dict:
            config_dict['rand_aug_mag'] = self.cfg.task.data.rand_aug_mag

        return config_dict

    def prepare_initial_ckpt(self, solution):
        model = instantiate(self.cfg.task.model)
        optimizer = get_optimizer(self.cfg, model, self.get_optimizer_vals(solution))
        scaler = GradScaler()
        return {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()}


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(images, labels, outs, class_names):
    preds = torch.argmax(outs, 1)
    preds = np.squeeze(preds.numpy())
    probs = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, outs)]
    # plot the images in the batch, along with predicted and true labels
    plt.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(12, 16))
    n_images = len(images)
    r = int(np.sqrt(n_images))
    c = int(np.ceil(n_images / r))
    for idx in range(n_images):
        ax = fig.add_subplot(r, c, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel='T-shirt/top' in class_names)
        ax.set_title(f"{class_names[preds[idx]]}, {probs[idx] * 100.0:.1f}\n({class_names[labels[idx]]})",
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    # make margins small
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    return fig
