# Learning rate scheduler
def lr_scheduler(optimizer, lr_mult, args, weight_mult=1, ):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult / 10.0
        else:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult
        counter += 1

    return optimizer, lr_mult


def dom_w_scheduler(optimizer, lr_mult, args, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult * weight_mult
        counter += 1

    return optimizer, lr_mult
