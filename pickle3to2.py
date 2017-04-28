import numpy as np
import pickle as pkl

###
# For Prince purposes : Easier to use Pytorch with Python 2.7
# Thus pickle protocole 3 do not work.
# Saving with numpy allows pickle readable with Python 2.7 and 3
###

train_X = pkl.load(open('data/train_X.pkl','rb'))
np.save('train_X', train_X)
train_y = pkl.load(open('data/train_y.pkl','rb'))
np.save('train_y',train_y)

val_X = pkl.load(open('data/val_X.pkl','rb'))
np.save('val_X',val_X)
val_y = pkl.load(open('data/val_y.pkl','rb'))
np.save('val_y',val_y)

test_X = pkl.load(open('data/test_X.pkl','rb'))
np.save('test_X',test_X)
test_y = pkl.load(open('data/test_y.pkl','rb'))
np.save('test_y',test_y)
