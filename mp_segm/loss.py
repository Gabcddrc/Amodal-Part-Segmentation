import torch
import torch.nn as nn
import dl
import jaccardLoss
import torch.nn.functional as F

class SegmLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.loss = DiceCELoss()
		# self.loss = jaccardLoss.JaccardLoss("multiclass")
		# self.loss = nn.CrossEntropyLoss(reduction="None")

	def forward(self, data_dict):
		segm_128 = data_dict['segm_128']

		segm_target_l = segm_128[:, 0]
		segm_target_r = segm_128[:, 1]

		segm_logits = data_dict['segm_logits']
		segm_logits_l, segm_logits_r = torch.split(segm_logits, 17, 1)

		# amodal loss
		dist_hand_l = self.loss(segm_logits_l, segm_target_l)
		dist_hand_r = self.loss(segm_logits_r, segm_target_r)
		total_loss = dist_hand_l.mean().view(-1) + dist_hand_r.mean().view(-1)
		return total_loss
		

def to_one_hot(tensor,nClasses):
	
	n,h,w = tensor.size()
	one_hot = torch.zeros(n,nClasses,h,w).scatter_(1,tensor.view(n,1,h,w),1)
	return one_hot

class mIoULoss(nn.Module):
	def __init__(self, weight=None, size_average=True, n_classes=2):
		super(mIoULoss, self).__init__()
		self.classes = n_classes

	def forward(self, inputs, target):
		# inputs => N x Classes x H x W
		# target_oneHot => N x Classes x H x W
		target_oneHot = to_one_hot(target, 17)
		N = inputs.size()[0]

		# predicted probabilities for each pixel along channel
		inputs = F.softmax(inputs,dim=1)
		
		# Numerator Product
		inter = inputs * target_oneHot
		## Sum over all pixels N x C x H x W => N x C
		inter = inter.view(N,self.classes,-1).sum(2)

		#Denominator 
		union= inputs + target_oneHot - (inputs*target_oneHot)
		## Sum over all pixels N x C x H x W => N x C
		union = union.view(N,self.classes,-1).sum(2)

		loss = inter/union

		## Return average loss over classes and batch
		return -loss.mean()

class DiceCELoss(nn.Module):
	def __init__(self):
		super(DiceCELoss, self).__init__()
		self.dice = dl.DiceLoss("multiclass")
		self.ce = nn.CrossEntropyLoss(reduction="none")

	def forward(self, inputs, targets):
		
		
		return self.dice(inputs, targets) + self.ce(inputs, targets)