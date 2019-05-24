# Introduction <img src="fairseq_logo.png" width="50"> 

-**Personal change:
  -**add a new model named len_pre_transformer.py. it takes advantages of transfomer's encoder and get rid of the decoder. as for predict sentence length, it stacks three fully connected layers above the encoder, with a 'mean' operation between the first two layers.
  -**add a new crition named acc_label_smooth.py. it add acc model in the crition.
  -**add 'train_acc' and 'valid_acc' in train.py and trainer.py. by doing this



- ** there are some redundant print needed to be deleted
- ** modified by linux 
- ** modified by windows

- ** modified in develop branch

switch origin branch: git branch -u origin/develop (or origin/hotfix or origin/master)

