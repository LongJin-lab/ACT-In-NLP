import torch
import torch.optim as optim

'''
your train code
'''


#original
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#ACT
optimizer = optim.SGD_ACT(model.parameters(), lr=args.lr, momentum=args.momentum, gamma=0.3, delta=4.5)


'''
your train code
'''


