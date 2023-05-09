import torch


def compute_scale_and_shift(prediction, target, mask):
    """
    Source: https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(prediction[mask] * prediction[mask])
    a_01 = torch.sum(prediction[mask])
    a_11 = torch.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(prediction[mask] * target[mask])
    b_1 = torch.sum(target[mask])

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1