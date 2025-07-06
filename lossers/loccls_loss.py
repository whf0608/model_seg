import lossers.lovasz_loss1 as L
from torch.nn import functional as F

def loccls_loss(rs,data,mask_key='label',n_classes=2):
    labels_clf = data[mask_key]
    labels_loc = labels_clf.clone()
    labels_loc[labels_loc>1]=1
    labels_loc = labels_loc.cuda().long()
    labels_clf = labels_clf.cuda().long()
    
    outout_loc, output_clf = rs
    
    ce_loss_loc = F.cross_entropy(outout_loc, labels_loc, ignore_index=255)
    lovasz_loss_loc = L.lovasz_softmax(F.softmax(outout_loc, dim=1), labels_loc, ignore=255)

    ce_loss_clf = F.cross_entropy(output_clf, labels_clf)
    lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)      
    final_loss = ce_loss_loc + ce_loss_clf + 0.75 * lovasz_loss_clf  + 0.5 * lovasz_loss_loc 

    return final_loss 


def loss_cross_entropy_lovasz(rs,data,mask_key='label',n_classes=2):
    mask = data[mask_key].cuda().long()

    ce_loss_clf = F.cross_entropy(rs[0],mask)
    lovasz_loss_clf = L.lovasz_softmax(F.softmax(rs[0], dim=1), mask.float(), ignore=None)      
    loss = ce_loss_clf + 0.75 * lovasz_loss_clf 
    
    return loss