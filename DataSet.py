import torch,cv2,os
import albumentations as A
class FootBallDataSet(torch.utils.data.Dataset):
    def __init__(self,images_dir:str,transforms:A.Compose=None)->None:
        super(FootBallDataSet,self).__init__()
        self.images_dir=images_dir
        self.transforms=transforms
        self.images=[]
        for root, _,files in os.walk(self.images_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") :
                    self.images.append(os.path.join(root,file))
        self.labels=dict(goals=0, red_card=1, tackles=2)
    def __len__(self)->int:
        return len(self.images)
    def __getitem__(self, idx:int)->dict:
        img=cv2.imread(filename=self.images[idx])
        img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        global label
        for k,v in self.labels.items():
            if k in self.images[idx]:
                label=self.labels[k]
                break
        if self.transforms:
            img=self.transforms(image=img)["image"]
        return dict(image=img,label=torch.tensor(data=label,dtype=torch.long))
"""


dataset=FootBallDataSet(images_dir="/home/muhammad/Downloads/football_match_dataset",transforms=transforms)
train_set_length=int(len(dataset)*0.8)
val_set_length=int(len(dataset)-train_set_length)
train_dataset,val_dataset=torch.utils.data.random_split(dataset=dataset,generator=torch.Generator().manual_seed(42),
                                                        lengths=[train_set_length,val_set_length])
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=32,shuffle=False)
for batch in val_loader:
    print(batch)
"""