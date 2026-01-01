import torch.optim as optim

def get_optimizer(config, parameters):
    lr = float(config.optim.lr)
    weight_decay = float(config.optim.weight_decay)
    
    if config.optim.optimizer == 'Adam':
        eps = float(config.optim.eps) 
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=eps)
                          
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
        
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
