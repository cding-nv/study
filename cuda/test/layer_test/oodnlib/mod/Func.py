import torch
import oodnlib.ext.func
from .Plugin import register_op


def AdjMatrixBatchSimpleGenerate(inp: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    register_op("func.AdjMatrixBatchSimpleGenerate")
    if inp.dim() == 1:
        inp = inp.unsqueeze(0)
    if inp.dtype != torch.float:
        inp = inp.to(torch.float)
    return torch.ops.func.AdjMatrixBatchSimpleGenerate(inp, alpha)


def BatchingSequenceOfSequenceDataReduceInvalidInput(inp: torch.Tensor, seq_len: int):
    # register_op("func.BatchingSequenceOfSequenceDataReduceInvalidInputA")
    # register_op("func.BatchingSequenceOfSequenceDataReduceInvalidInputB")
    length = torch.ops.func.BatchingSequenceOfSequenceDataReduceInvalidInputA(
        inp, seq_len
    )
    length_sum = length[0].sum().item()
    output = torch.ops.func.BatchingSequenceOfSequenceDataReduceInvalidInputB(
        inp, length, length_sum, seq_len
    )
    return output, inp.size(0), length


def PositionsAndTimeDiff(time: torch.Tensor, mask: torch.Tensor):
    # register_op("func.PositionsAndTimeDiff")
    return torch.ops.func.PositionsAndTimeDiff(time, mask)


def RecoverSequenceOfSequenceDataReduceInvalidInput(
    inp: torch.Tensor, length: torch.Tensor, seq_len: int
) -> torch.Tensor:
    dim = inp.dim()
    if dim == 1:
        if inp.requires_grad:
            return (
                torch.ops.func.RecoverSequenceOfSequenceDataReduceInvalidInput1D_train(
                    inp, length, seq_len
                )
            )
        else:
            # register_op("func.RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer")
            return (
                torch.ops.func.RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer(
                    inp, length, seq_len
                )
            )
    elif dim == 2:
        if inp.requires_grad:
            return (
                torch.ops.func.RecoverSequenceOfSequenceDataReduceInvalidInput2D_train(
                    inp, length, seq_len
                )
            )
        else:
            # register_op("func.RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer")
            return (
                torch.ops.func.RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer(
                    inp, length, seq_len
                )
            )
    else:
        return None
