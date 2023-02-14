"""
Script containing various utilities related to model training, testing, and extraction of attention
weights.
"""

import logging
from typing import Callable, Iterable
import tensorboard
import torch.nn as nn
import torch
from utils.data_utils import get_features
import torch.nn.functional as F
import numpy as np
from .logger import MetricLogger

from .output import ansi, get_ansi_len, output_iter, prints
from torch.utils.tensorboard import SummaryWriter

@torch.no_grad()
def save_fn( file_path: str = None, folder_path: str = None,
            suffix: str = None, component: str = '',
            verbose: bool = False, indent: int = 0, **kwargs):
    r"""Save pretrained model weights.

    Args:
        file_path (str | None): The file path to save pretrained weights.
            Defaults to ``'{folder_path}/{self.name}{suffix}.pth'``.
        folder_path (str | None): The folder path containing model checkpoint.
            It is used when :attr:`file_path` is not provided.
            Defaults to :attr:`self.folder_path`.
        suffix (str | None): The suffix string to model weights file.
            Defaults to :attr:`self.suffix`.
        component (str): Specify which part of the weights to save.
            Choose from ``['full', 'features', 'classifier']``.
            Defaults to ``'full'``.
        verbose (bool): Whether to output auxiliary information.
            Defaults to ``False``.
        indent (int): The indent of output auxialiary information.
        **kwargs: Keyword arguments passed to :any:`torch.save`.
    """
    if file_path is None:
        folder_path = folder_path if folder_path is not None \
            else self.folder_path
        suffix = suffix if suffix is not None else self.suffix
        file_path = os.path.normpath(os.path.join(
            folder_path, f'{self.name}{suffix}.pth'))
    else:
        folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # TODO: type annotation might change? dict[str, torch.Tensor]
    module = self._model
    if component == 'features':
        module = self._model.features
    elif component == 'classifier':
        module = self._model.classifier
    else:
        assert component == '', f'{component=}'
    _dict: OrderedDict[str, torch.Tensor] = module.state_dict(
        prefix=component)
    torch.save(_dict, file_path, **kwargs)
    if verbose:
        prints(
            f'Model {self.name} saved at: {file_path}', indent=indent)
        
def interleave_fn(self, xy, batch):
    nu = len(xy) - 1
    offsets = self.interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
    
def activate_params(module: nn.Module, params: Iterator[nn.Parameter] = []):
    r"""Set ``requires_grad=True`` for selected :attr:`params` of :attr:`module`.
    All other params are frozen.

    Args:
        module (torch.nn.Module): The module to process.
        params (~collections.abc.Iterator[torch.nn.parameter.Parameter]):
            The parameters to ``requires_grad``.
                Defaults to ``[]``.
    """
    module.requires_grad_(False)
    for param in params:
        param.requires_grad_()


@torch.no_grad()
def accuracy_fn(_output: torch.Tensor, _label: torch.Tensor, num_classes: int,
             topk: Iterable[int] = (1, 5)) -> list[float]:
    r"""Computes the accuracy over the k top predictions
    for the specified values of k.

    Args:
        _output (torch.Tensor): The batched logit tensor with shape ``(N, C)``.
        _label (torch.Tensor): The batched label tensor with shape ``(N)``.
        num_classes (int): Number of classes.
        topk (~collections.abc.Iterable[int]): Which top-k accuracies to show.
            Defaults to ``(1, 5)``.

    Returns:
        list[float]: Top-k accuracies.
    """
    maxk = min(max(topk), num_classes)
    batch_size = _label.size(0)
    _, pred = _output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(_label[None])
    res: list[float] = []
    for k in topk:
        if k > num_classes:
            res.append(100.0)
        else:
            correct_k = float(correct[:k].sum(dtype=torch.float32))
            res.append(correct_k * (100.0 / batch_size))
    return res

def val_loss(_input: torch.Tensor = None, _label: torch.Tensor = None,
            _output: torch.Tensor = None, reduction: str = 'mean', **kwargs) -> torch.Tensor:
    
    criterion = nn.CrossEntropyLoss(reduction=reduction)

    return criterion(_output, _label)


def linear_rampup(iter, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(iter / rampup_length, 0.0, 1.0)
        return float(current)

def SemiLoss(outputs_x, targets_x, outputs_u, targets_u, iter):
    # Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.train_iteration)
    # lx is cross entropy, lu is L2 normalization
    
    probs_u = torch.softmax(outputs_u, dim=1)
    Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    Lu = torch.mean((probs_u - targets_u)**2)
    lambda_u = 100.0
    w = lambda_u * linear_rampup(iter,16000)
    loss = Lx + w * Lu
    
    return loss


def loss_fn(_input: torch.Tensor = None, _label: torch.Tensor = None,
            _output: torch.Tensor = None, reduction: str = 'mean',_soft_label: torch.Tensor = None,
            temp: float = 1.0, outputs_x=None, targets_x=None, outputs_u=None, targets_u=None, iter=None,
            **kwargs) -> torch.Tensor:
    r"""Calculate the loss using :attr:`self.criterion`
    (:attr:`self.criterion_noreduction`).

    Args:
        _input (torch.Tensor | None): The batched input tensor.
            If :attr:`_output` is provided, this argument will be ignored.
            Defaults to ``None``.
        _label (torch.Tensor): The label of the batch with shape ``(N)``.
        _output (torch.Tensor | None): The logits of :attr:`_input`.
            If ``None``, use :attr:`_input` to calculate logits.
            Defaults to ``None``.
        reduction (str): Specifies the reduction to apply to the output.
            Choose from ``['none', 'mean']``.
            Defaults to ``'mean'``.
        **kwargs: Keyword arguments passed to :meth:`get_logits()`
            if :attr:`_output` is not provided.

    Returns:
        torch.Tensor:
            A scalar loss tensor (with shape ``(N)`` if ``reduction='none'``).
    """
    if outputs_x is not None:
        return SemiLoss(outputs_x, targets_x, outputs_u, targets_u, iter)
        
    if _soft_label is None:
        return val_loss(_input=_input, _label=_label, _output=_output, reduction=reduction)
        
    criterion = nn.CrossEntropyLoss(reduction=reduction)

    return criterion(_output,_soft_label)




def distillation(module: nn.Module, num_classes: int,
          epochs: int, optimizer, lr_scheduler,
        log_dir:str = 'runs/test', 
          grad_clip: float = 5.0, 
          print_prefix: str = 'Distill', start_epoch: int = 0, resume: int = 0,
          validate_interval: int = 1, save: bool = False,
          loader_train: torch.utils.data.DataLoader = None,
          loader_valid: torch.utils.data.DataLoader = None,
          epoch_fn: Callable[..., None] = None,
        file_path: str = None,
          folder_path: str = None, suffix: str = None,
          writer=None, main_tag: str = 'train', tag: str = '',
          accuracy_fn: Callable[..., list[float]] = None,
          verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
          change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
          backward_and_step: bool = True, 
          mixmatch: bool = False,
          **kwargs):
    r"""Train the model"""
    if epochs <= 0:
        return
    get_data_fn = get_data_fn or (lambda x: x)
    forward_fn = forward_fn or module.__call__



    writer = SummaryWriter(log_dir=log_dir)
    validate_fn = dis_validate 


    scaler: torch.cuda.amp.GradScaler = None

    best_validate_result = (0.0, float('inf'))
    if validate_interval != 0:
        best_validate_result = validate_fn(loader=loader_valid, 
                                           forward_fn=forward_fn, loss_fn=loss_fn,
                                           writer=None, tag=tag, _epoch=start_epoch,
                                           verbose=verbose, indent=indent,  **kwargs)
        best_acc = best_validate_result[0]

    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])
    len_loader_train = len(loader_train)
    total_iter = (epochs - resume) * len_loader_train

    logger = MetricLogger()
    if mixmatch:
        logger.create_meters(loss=None)
    else:
        logger.create_meters(  gt_loss=None, gt_acc1=None, gt_acc5=None, 
                          hapi_loss=None, hapi_acc1=None, hapi_acc5=None)
    if resume and lr_scheduler:
        for _ in range(resume):
            lr_scheduler.step()
    iterator = range(resume, epochs)
    if verbose and output_freq == 'epoch':
        header: str = '{blue_light}{0}: {reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(header), 30) + get_ansi_len(header))
        iterator = logger.log_every(range(resume, epochs),
                                    header=print_prefix,
                                    tqdm_header='Epoch',
                                    indent=indent)
    for _epoch in iterator:
        _epoch += 1
        logger.reset()
        if callable(epoch_fn):
            activate_params(module, [])
            epoch_fn(optimizer=optimizer, lr_scheduler=lr_scheduler,
                     _epoch=_epoch, epochs=epochs, start_epoch=start_epoch)
        loader_epoch = loader_train
        if verbose and output_freq == 'iter':
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader_epoch = logger.log_every(loader_train, header=header,
                                            tqdm_header='Batch',
                                            indent=indent)
        if change_train_eval:
            module.train()
        activate_params(module, params)




        if _epoch < 10000:
            mode = 'train_STU' #kl loss / return raw data
            print(_epoch,mode)
        elif _epoch >= 10000:
            mode = 'train_ADV_STU'  #kl loss / return adv data
            print(_epoch,mode)


        for i, data in enumerate(loader_epoch):
            _iter = _epoch * len_loader_train + i

            if mixmatch:
                print('no mixmatch support') 
                # input_ids, token_type_ids, attention_mask, label, soft_label, hapi_label  = data


                # mixed_input = list(torch.split(mixed_input, batch_size))
                # mixed_input = interleave_fn(mixed_input, batch_size)

                # logits = [forward_fn(mixed_input[0])]
                # for input in mixed_input[1:]:
                #     logits.append(forward_fn(input))

                # # put interleaved samples back
                # logits = interleave_fn(logits, batch_size)
                # logits_x = logits[0]
                # logits_u = torch.cat(logits[1:], dim=0)

                # loss = loss_fn(outputs_x = logits_x, targets_x = mixed_target[:batch_size], outputs_u = logits_u, targets_u = mixed_target[batch_size:], iter = _iter)


            else:
                input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                

                _output = forward_fn(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
                loss = loss_fn( _soft_label=_soft_label, _output=_output)

            if backward_and_step:
                optimizer.zero_grad()
                #backward the weights 
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

          

            if lr_scheduler and lr_scheduler_freq == 'iter':
                lr_scheduler.step()
                
               
            if mixmatch:
                 logger.update(n=batch_size, loss=float(loss))
            else:    
                hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=num_classes, topk=(1, 5))
                gt_acc1, gt_acc5 = accuracy_fn(
                    _output, _label, num_classes=num_classes, topk=(1, 5))
                batch_size = int(_label.size(0)) 
                logger.update(n=batch_size, gt_acc1=gt_acc1, gt_acc5=gt_acc5, 
                            hapi_loss=float(loss), hapi_acc1=hapi_acc1, hapi_acc5=hapi_acc5)
        optimizer.zero_grad()
        if lr_scheduler and lr_scheduler_freq == 'epoch':
            lr_scheduler.step()
        if change_train_eval:
            module.eval()
        activate_params(module, [])
        if mixmatch:
            loss=(logger.meters['loss'].global_avg)
            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                writer.add_scalars(main_tag='loss/' + main_tag,
                            tag_scalar_dict={tag: loss}, global_step=_epoch + start_epoch)        
        else:
            gt_acc1, hapi_loss, hapi_acc1 = (
                    logger.meters['gt_acc1'].global_avg,
                    logger.meters['hapi_loss'].global_avg,
                    logger.meters['hapi_acc1'].global_avg)
            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                            tag_scalar_dict={tag: gt_acc1}, global_step=_epoch + start_epoch)        
                writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                            tag_scalar_dict={tag: hapi_loss}, global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                        tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch + start_epoch)
            
        if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
            validate_result = validate_fn(module=module,
                                          num_classes=num_classes,
                                          loader=loader_valid,
                                          writer=writer, tag=tag,
                                          _epoch=_epoch + start_epoch,
                                          verbose=verbose, indent=indent,
                                          **kwargs)
            cur_acc = validate_result[0]
            if cur_acc >= best_acc:
                best_validate_result = validate_result
                if verbose:
                    prints('{purple}best result update!{reset}'.format(
                        **ansi), indent=indent)
                    prints(f'Current Acc: {cur_acc:.3f}    '
                           f'Previous Best Acc: {best_acc:.3f}',
                           indent=indent)
                best_acc = cur_acc
                if save:
                    save_fn(file_path=file_path, folder_path=folder_path,
                            suffix=suffix, verbose=verbose)
            if verbose:
                prints('-' * 50, indent=indent)
    module.zero_grad()
    return best_validate_result


def dis_validate(module: nn.Module, num_classes: int,
             loader: torch.utils.data.DataLoader,
             print_prefix: str = 'Validate', indent: int = 0,
             verbose: bool = True,
            
           
             loss_fn: Callable[..., torch.Tensor] = None,
             writer=None, main_tag: str = 'valid',
             tag: str = '', _epoch: int = None,
             accuracy_fn: Callable[..., list[float]] = None,
                        stu_arch_list=None,
              tea_forward_fn: Callable[..., torch.Tensor] = None,
             **kwargs) -> tuple[float, float]:
    r"""Evaluate the model.

    Returns:
        (float, float): Accuracy and loss.
    """
    module.eval()
   
    forward_fn =  module.__call__

    logger = MetricLogger()
    logger.create_meters( gt_loss=None, gt_acc1=None, gt_acc5=None, 
                          hapi_loss=None, hapi_acc1=None, hapi_acc5=None)
    loader_epoch = loader  
    if verbose:
        header: str = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
        loader_epoch = logger.log_every(loader, header=header,
                                        tqdm_header='Batch',
                                        indent=indent)
    for data in loader_epoch:

        input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                
        _output = forward_fn(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        loss = loss_fn( _soft_label=_soft_label, _output=_output)
                
        with torch.no_grad():
            _output = forward_fn(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
            gt_loss = float(loss_fn( _label=_label, _output=_output, **kwargs))
            hapi_loss = float(loss_fn( _soft_label=_soft_label, _output=_output, temp=1.0, **kwargs))
            hapi_acc1, hapi_acc5 = accuracy_fn(
                _output, hapi_label, num_classes=num_classes, topk=(1, 5))
            gt_acc1, gt_acc5 = accuracy_fn(
                _output, _label, num_classes=num_classes, topk=(1, 5))
            batch_size = int(_label.size(0))
            logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1, gt_acc5=gt_acc5, 
                          hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1, hapi_acc5=hapi_acc5)
            
    gt_loss, gt_acc1, hapi_loss, hapi_acc1 = (logger.meters['gt_loss'].global_avg,
                 logger.meters['gt_acc1'].global_avg,
                 logger.meters['hapi_loss'].global_avg,
                 logger.meters['hapi_acc1'].global_avg)


    if writer is not None and _epoch is not None and main_tag:
        from torch.utils.tensorboard import SummaryWriter
        assert isinstance(writer, SummaryWriter)
        writer.add_scalars(main_tag='gt_loss/' + main_tag,
                    tag_scalar_dict={tag: gt_loss}, global_step=_epoch)
        writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                    tag_scalar_dict={tag: gt_acc1}, global_step=_epoch)        
        writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                    tag_scalar_dict={tag: hapi_loss}, global_step=_epoch)
        writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                    tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch)

    return hapi_acc1, hapi_loss

