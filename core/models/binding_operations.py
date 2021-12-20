import torch
import torch.nn as nn

class BindingLayer:

	def for_type(binder: str):
		if binder == "tpr":
			return SumFlattenedOuterProduct()
		else:
			raise NotImplementedError

class SumFlattenedOuterProduct(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, input):
		input1, input2 = input

		if len(input1.shape) < 3: # if batch has only one (squeezed) entry
			input1 = input1.unsqueeze(0)
			input2 = input2.unsqueeze(0)

		outer_prod = torch.bmm(input1.transpose(1,2), input2)
		flat_outer_prod = outer_prod.view(outer_prod.size()[0], -1).unsqueeze(0)
		sum_flat_outer_prod = flat_outer_prod
		return sum_flat_outer_prod
