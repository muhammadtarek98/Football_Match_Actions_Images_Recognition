import torch,torchvision,torchinfo
class MLP(torch.nn.Module):
    def __init__(self,in_features:int,out_features:int,dropout_rate:float=0.0)->None:
        super(MLP,self).__init__()
        self.fc=torch.nn.Linear(in_features=in_features,out_features=out_features)
        self.bn=torch.nn.BatchNorm1d(num_features=out_features)
        self.act=torch.nn.LeakyReLU(inplace=True,negative_slope=0.02)
        self.dropout=torch.nn.Dropout(p=dropout_rate) if dropout_rate>0.0 else torch.nn.Identity()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.fc(x)
        x=self.bn(x)
        x=self.act(x)
        x=self.dropout(x)
        return x

class Model(torch.nn.Module):
    def __init__(self,num_classes:int)->None:
        super(Model,self).__init__()
        self.backbone=torchvision.models.vgg16_bn(pretrained=True,progress=True)
        self.out_channels=self.backbone.features[-3].num_features
        self.feature_extractor=torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.flatten=torch.nn.AdaptiveAvgPool2d((1,1))
        self.mlp_block=MLP(in_features=self.out_channels,out_features=self.out_channels//2)
        self.mlp_block_2=MLP(in_features=self.out_channels//2,out_features=self.out_channels//4)
        self.fc=torch.nn.Linear(in_features=self.out_channels//4,out_features=num_classes)
        #self.act=torch.nn.Softmax(dim=1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        features=self.feature_extractor(x)
        flatten_feature_map=self.flatten(features)
        flatten_feature_map=torch.flatten(flatten_feature_map,start_dim=1)
        feature_vector=self.mlp_block(flatten_feature_map)
        feature_vector=self.mlp_block_2(feature_vector)
        feature_vector=self.fc(feature_vector)
        return feature_vector
