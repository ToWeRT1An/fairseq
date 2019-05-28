from os import walk
from PIL import Image
import sys, getopt

def get_argv(argv):
	inputfold = ''
	outputfile = ''

	try:
		opts,args = getopt.getopt(argv,"hi:o:",['ifold=','ofile='])
	except getopt.GetoptError:
		print('parse_image.py -i <inputfold> -o <outputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('parse_image.py -i <inputfold> -o <outputfile>')
		elif opt in ('-i','--ifold'):
			inputfold = arg
		elif opt in ('-o','--ofile'):
			outputfile = arg
	print('inputfold is ',inputfold)
	print('outputfile is ',outputfile)

	return inputfold, outputfile

inputfold, outputfile = get_argv(sys.argv[1:])

images = []
for (dirpath, dirnames, filenames) in walk(inputfold):
	images.extend(filenames)
	break
images.sort(key = lambda x: x)
print(images)
import torch 
from torchvision import transforms

import matplotlib.pyplot as pyplot
import json

json_file = open(outputfile,'w')
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



