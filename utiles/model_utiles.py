
from thop import profile
import torchprof
from fvcore.nn import FlopCountAnalysis, parameter_count_table,flop_count_str,flop_count_table,ActivationCountAnalysis

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def analysis_model(model,x=None):
    # fa = FlopCountAnalysis(model, imgs)
    # print(flop_count_str(FlopCountAnalysis(model, (imgs,imgs))))
    tabel = flop_count_table(FlopCountAnalysis(model, x))
    # acts = ActivationCountAnalysis(model, imgs)
    # acts.by_module()
    return tabel

def analysis_model1(model,x=None):
    flops, params = profile(model, (imgs,))
    with torchprof.Profile(model, use_cuda=False) as prof:
        model(imgs)
    trace, event_lists_dict  = prof.raw()
    return trace, event_lists_dict

# imgs = torch.ones((1,3,640,640))
# print(analysis_model(model,x=(imgs,imgs)))