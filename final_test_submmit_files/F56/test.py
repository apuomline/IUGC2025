
import timm
import torch
if __name__=='__main__':

    x = torch.rand(1,3,224,224)
   
    model = timm.create_model(
        model_name=   'levit_256.fb_dist_in1k',
        pretrained=False,
  in_chans=3, num_classes=0, global_pool='', features_only=True
)
    outs = model(x)
    for fea in outs:
        print(fea.shape)