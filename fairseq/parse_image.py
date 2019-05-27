from os import walk
from PIL import Image
images = []
for (dirpath, dirnames, filenames) in walk('./'):
	images.extend(filenames)
	break
images.sort(key = lambda x: x)
print(images)
import torch 
from torchvision import transforms

import matplotlib.pyplot as pyplot
import json

json_file = open('len_label.json','w')
trans = transforms.ToTensor()
JSON = {}
id = 0
for img in images:
    Img = Image.open(img)
    attn = trans(Img).squeeze(0)
    print(attn.shape)
    print(attn)
    values, indices = torch.topk(attn,1)
    label = torch.zeros(attn.shape)
    label = label.scatter_(1,indices,1).sum(dim=0)

    time, bach_size, tgt_length, src_length, seri_num = img.split('_')

    json_content ={'bach_size':bach_size,'tgt_length':tgt_length,
				'src_length':src_length,'seri_num':seri_num,
				'respose_pos':indices.numpy().tolist(),
				'word_trans':label.numpy().tolist()}

    JSON[id]=json_content
    id += 1
json.dump(JSON,json_file)

json_file.close()



