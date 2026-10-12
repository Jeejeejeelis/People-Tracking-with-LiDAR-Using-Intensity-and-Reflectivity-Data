import torch

try:
    from . import ingroup_inds_cuda
    ingroup_indices = ingroup_inds_cuda
except ImportError:
    ingroup_indices = None
    # print('Can not import ingroup indices')

from torch.autograd import Function
class IngroupIndicesFunction(Function):

    @staticmethod
    def forward(ctx, group_inds):
        if ingroup_indices is None:
             # Fallback or error if actually called, but PointPillars shouldn't call this
             raise ImportError("ingroup_inds_cuda not compiled.")

        out_inds = torch.zeros_like(group_inds) - 1

        ingroup_indices.forward(group_inds, out_inds)

        ctx.mark_non_differentiable(out_inds)

        return out_inds

    @staticmethod
    def backward(ctx, g):

        return None

ingroup_inds = IngroupIndicesFunction.apply
