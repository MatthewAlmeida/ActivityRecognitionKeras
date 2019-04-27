import math

"""
This file contains functions that take in the total
number of training epochs, the maximum learning rate,
and the tail length (the duration of the rapid 
decrease in LR at the end of training in the 1cycle
policy) and return new functions that take an epoch
index as a parameter and return the proper LR and
momentum for that epoch.

These functions are handed to the scheduler objects
that keras uses to set the optimizer LR / Momentum
values during training, via callbacks.
"""

def get_one_cycle_lr_fn(total_steps, max_lr, tail_len):
    if tail_len >= total_steps:
        raise ValueError("Total number of steps must be longer than the tail.")

    cycle_len = total_steps - tail_len
    max_lr_step = math.floor(cycle_len / 2)
    initial_lr = max_lr / 10
    final_lr = max_lr / 1000    
    
    linear_change = (max_lr - initial_lr) / max_lr_step
    tail_linear_change = (final_lr - initial_lr) / tail_len
    
    neg_intercept = max_lr + (linear_change * max_lr_step)
    tail_intercept = initial_lr - (tail_linear_change * cycle_len)
    
    def one_cycle_fn(epoch):
        if epoch <= max_lr_step:
            lr = linear_change * epoch + initial_lr
        elif epoch > max_lr_step and epoch <= cycle_len:
            lr = -1 * linear_change * epoch + neg_intercept
        else:
            lr = tail_linear_change * epoch + tail_intercept
            
        return lr
    
    return one_cycle_fn  

def get_one_cycle_momentum_fn(total_steps, max_lr, tail_len, max_momentum=0.95, min_momentum=0.85):
    if tail_len >= total_steps:
        raise ValueError("Total number of steps must be longer than the tail.")
        
    if min_momentum >= max_momentum:
        raise ValueError("Maximum momentum is less than minimum momentum.")
    
    cycle_len = total_steps - tail_len
    max_lr_step = math.floor(cycle_len / 2)

    momentum_linear_change = (max_momentum - min_momentum) / max_lr_step
    momentum_rising_intercept = min_momentum - momentum_linear_change * max_lr_step
    
    def one_cycle_mom_fn(epoch):
        if epoch <= max_lr_step:
            mom = -1 * momentum_linear_change * epoch + max_momentum
        elif epoch > max_lr_step and epoch <= cycle_len:
            mom = (momentum_linear_change * epoch) + momentum_rising_intercept
        else:
            mom = max_momentum
            
        return mom
    
    return one_cycle_mom_fn