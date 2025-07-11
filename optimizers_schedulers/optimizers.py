import logging

from torch.optim import ASGD, Adamax, Adadelta, Adagrad,\
    Adam,AdamW, SGD, SparseAdam,RMSprop,NAdam, RAdam, Rprop

optimizer_dic = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
    'adamw': AdamW,
    'sparseadam': SparseAdam,
    'nadam': NAdam,
    'radam': RAdam,
    'rprop': Rprop

}
def get_optimizer(opt_name):
    if opt_name is None:

        return SGD

    else:
        if opt_name not in optimizer_dic:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))
        return optimizer_dic[opt_name]
