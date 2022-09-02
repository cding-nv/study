#include "AdjMatrixBatch.h"
#include "BatchingSequence.h"
#include "PositionsAndTimeDiff.h"
#include "RecoverSequenceInfer.h"
#include "RecoverSequenceTrain.h"
#include "errMsg.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
using torch::autograd::tensor_list;


torch::Tensor
AdjMatrixBatchSimpleGenerate(
        const torch::Tensor& inp,
        double alpha) {
    CHECK_INPUT(inp);
    const int batch_size = inp.size(0);
    const int seq_len = inp.size(1);
    torch::Tensor out = torch::empty({batch_size, seq_len, seq_len}, torch::dtype(torch::kFloat32).device(inp.device()));

    if (inp.scalar_type() == at::ScalarType::Float) {
        launchAdjMatrixBatch<float>(
                at::cuda::getCurrentCUDAStream(),
                (const float*)inp.data_ptr(),
                (float*)out.data_ptr(),
                batch_size,
                seq_len,
                alpha);
    }
    else {
        errMsg("invalid dtype");
    }

    return out;
}


torch::Tensor
BatchingSequenceOfSequenceDataReduceInvalidInputA(
        const torch::Tensor& inp,
        int64_t seq_len) {
    CHECK_INPUT(inp);
    const int batch_size = inp.size(0);
    const int sub_len = inp.size(1) / seq_len;
    torch::Tensor length = torch::empty({2, batch_size}, torch::dtype(torch::kInt64).device(inp.device()));

    if (inp.scalar_type() == at::ScalarType::Long) {
        launchMaskLength<int64_t>(
                at::cuda::getCurrentCUDAStream(),
                (const int64_t*)inp.data_ptr(),
                (int64_t*)length.data_ptr(),
                batch_size,
                seq_len,
                sub_len);
    }
    else {
        errMsg("invalid dtype");
    }

    return length;
}


torch::Tensor
BatchingSequenceOfSequenceDataReduceInvalidInputB(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t length_sum,
        int64_t seq_len) {
    CHECK_INPUT(inp);
    CHECK_INPUT(length);
    const int batch_size = inp.size(0);
    const int sub_len = inp.size(1) / seq_len;
    torch::Tensor out = torch::empty({length_sum, sub_len}, torch::dtype(inp.dtype()).device(inp.device()));

    if (inp.scalar_type() == at::ScalarType::Long) {
        launchBatchingSequence<int64_t>(
                at::cuda::getCurrentCUDAStream(),
                (const int64_t*)inp.data_ptr(),
                (int64_t*)length.data_ptr(),
                (int64_t*)out.data_ptr(),
                batch_size,
                seq_len,
                sub_len);
    }
    else {
        errMsg("invalid dtype");
    }

    return out;
}


std::vector<torch::Tensor>
PositionsAndTimeDiff(
        const torch::Tensor& time,
        const torch::Tensor& mask) {
    CHECK_INPUT(time);
    CHECK_INPUT(mask);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(time.device());

    const int64_t nRow = time.size(0);
    const int64_t nCol = time.size(1);
    const int64_t buff_size = time.numel() * 2;

    torch::Tensor buff = torch::empty(buff_size, options);
    torch::Tensor pos = torch::empty_like(time);
    torch::Tensor diff = torch::zeros_like(time);       // set first timediff to 0

    PositionsAndTimeDiff(
            at::cuda::getCurrentCUDAStream(),
            (const int64_t*)time.data_ptr(),
            (const int64_t*)mask.data_ptr(),
            (int64_t*)pos.data_ptr(),
            (int64_t*)diff.data_ptr(),
            (int64_t*)buff.data_ptr(),
            nRow,
            nCol);

    return {pos, diff};
}


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len) {
    CHECK_INPUT(inp);
    CHECK_INPUT(length);
    const int batch_size = length.size(1);
    torch::Tensor out = torch::zeros({batch_size, seq_len}, torch::dtype(inp.dtype()).device(inp.device()));

    switch (inp.scalar_type()) {
        case at::ScalarType::Half:
            RecoverSequenceInfer::forward1D<__half>(
                    at::cuda::getCurrentCUDAStream(),
                    (const __half*)inp.data_ptr(),
                    (const int64_t*)length.data_ptr(),
                    (__half*)out.data_ptr(),
                    batch_size,
                    seq_len);
            break;
        case at::ScalarType::Float:
            RecoverSequenceInfer::forward1D<float>(
                    at::cuda::getCurrentCUDAStream(),
                    (const float*)inp.data_ptr(),
                    (const int64_t*)length.data_ptr(),
                    (float*)out.data_ptr(),
                    batch_size,
                    seq_len);
            break;
        case at::ScalarType::Long:
            RecoverSequenceInfer::forward1D<int64_t>(
                    at::cuda::getCurrentCUDAStream(),
                    (const int64_t*)inp.data_ptr(),
                    (const int64_t*)length.data_ptr(),
                    (int64_t*)out.data_ptr(),
                    batch_size,
                    seq_len);
            break;
        default:
            errMsg("invalid dtype");
    }

    return out;
}


class RecoverSequenceAutograd1D: public torch::autograd::Function<RecoverSequenceAutograd1D> {
public:
    static torch::Tensor
    forward(torch::autograd::AutogradContext* ctx,
            const torch::Tensor& inp,
            const torch::Tensor& length,
            int64_t seq_len) {
        CHECK_INPUT(inp);
        CHECK_INPUT(length);
        const int batch_size = length.size(1);
        torch::Tensor out = torch::zeros({batch_size, seq_len}, torch::dtype(inp.dtype()).device(inp.device()));

        switch (inp.scalar_type()) {
            case at::ScalarType::Half:
                RecoverSequenceTrain::forward1D<__half>(
                        (const __half*)inp.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (__half*)out.data_ptr(),
                        batch_size,
                        seq_len,
                        at::cuda::getCurrentCUDAStream());
                break;
            case at::ScalarType::Float:
                RecoverSequenceTrain::forward1D<float>(
                        (const float*)inp.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (float*)out.data_ptr(),
                        batch_size,
                        seq_len,
                        at::cuda::getCurrentCUDAStream());
                break;
            default:
                errMsg("invalid dtype");
        }

        ctx->saved_data["length_sum"] = inp.size(0);
        ctx->save_for_backward({length});
        return out;
    }

    static tensor_list
    backward(torch::autograd::AutogradContext* ctx,
             const tensor_list& grad_list) {
        auto grad = grad_list[0];
        CHECK_INPUT(grad);

        auto length_sum = ctx->saved_data["length_sum"].toInt();
        ctx->saved_data.clear();
        tensor_list saved_list = std::move(ctx->get_saved_variables());
        torch::Tensor& length = saved_list[0];

        const int batch_size = grad.size(0);
        const int seq_len = grad.size(1);
        torch::Tensor dInp = torch::empty(length_sum, torch::dtype(grad.dtype()).device(grad.device()));

        switch (grad.scalar_type()) {
            case at::ScalarType::Half:
                RecoverSequenceTrain::backward1D<__half>(
                        (const __half*)grad.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (__half*)dInp.data_ptr(),
                        batch_size,
                        seq_len,
                        at::cuda::getCurrentCUDAStream());
                break;
            case at::ScalarType::Float:
                RecoverSequenceTrain::backward1D<float>(
                        (const float*)grad.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (float*)dInp.data_ptr(),
                        batch_size,
                        seq_len,
                        at::cuda::getCurrentCUDAStream());
                break;
            default:
                errMsg("invalid dtype");
        }

        return {dInp,
                torch::Tensor(),
                torch::Tensor()};
    }
};


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput1D_train(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len) {
    return RecoverSequenceAutograd1D::apply(
            inp,
            length,
            seq_len);
}


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len) {
    CHECK_INPUT(inp);
    CHECK_INPUT(length);
    const int batch_size = length.size(1);
    const int hidden_size = inp.size(-1);
    torch::Tensor out = torch::zeros({batch_size, seq_len, hidden_size}, torch::dtype(inp.dtype()).device(inp.device()));

    switch (inp.scalar_type()) {
        case at::ScalarType::Half:
            RecoverSequenceInfer::forward2D<__half>(
                    at::cuda::getCurrentCUDAStream(),
                    (const __half*)inp.data_ptr(),
                    (const int64_t*)length.data_ptr(),
                    (__half*)out.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size);
            break;
        case at::ScalarType::Float:
            RecoverSequenceInfer::forward2D<float>(
                    at::cuda::getCurrentCUDAStream(),
                    (const float*)inp.data_ptr(),
                    (const int64_t*)length.data_ptr(),
                    (float*)out.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size);
            break;
        case at::ScalarType::Long:
            RecoverSequenceInfer::forward2D<int64_t>(
                    at::cuda::getCurrentCUDAStream(),
                    (const int64_t*)inp.data_ptr(),
                    (const int64_t*)length.data_ptr(),
                    (int64_t*)out.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size);
            break;
        default:
            errMsg("invalid dtype");
    }

    return out;
}


class RecoverSequenceAutograd2D: public torch::autograd::Function<RecoverSequenceAutograd2D> {
public:
    static torch::Tensor
    forward(torch::autograd::AutogradContext* ctx,
            const torch::Tensor& inp,
            const torch::Tensor& length,
            int64_t seq_len) {
        CHECK_INPUT(inp);
        CHECK_INPUT(length);
        const int batch_size = length.size(1);
        const int hidden_size = inp.size(-1);
        torch::Tensor out = torch::zeros({batch_size, seq_len, hidden_size}, torch::dtype(inp.dtype()).device(inp.device()));

        switch (inp.scalar_type()) {
            case at::ScalarType::Half:
                RecoverSequenceTrain::forward2D<__half>(
                        (const __half*)inp.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (__half*)out.data_ptr(),
                        batch_size,
                        seq_len,
                        hidden_size,
                        at::cuda::getCurrentCUDAStream());
                break;
            case at::ScalarType::Float:
                RecoverSequenceTrain::forward2D<float>(
                        (const float*)inp.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (float*)out.data_ptr(),
                        batch_size,
                        seq_len,
                        hidden_size,
                        at::cuda::getCurrentCUDAStream());
                break;
            default:
                errMsg("invalid dtype");
        }

        ctx->saved_data["length_sum"] = inp.size(0);
        ctx->save_for_backward({length});
        return out;
    }

    static tensor_list
    backward(torch::autograd::AutogradContext* ctx,
             const tensor_list& grad_list) {
        auto grad = grad_list[0];
        CHECK_INPUT(grad);

        auto length_sum = ctx->saved_data["length_sum"].toInt();
        ctx->saved_data.clear();
        tensor_list saved_list = std::move(ctx->get_saved_variables());
        torch::Tensor& length = saved_list[0];

        const int batch_size = grad.size(0);
        const int seq_len = grad.size(1);
        const int hidden_size = grad.size(2);
        torch::Tensor dInp = torch::empty({length_sum, hidden_size}, torch::dtype(grad.dtype()).device(grad.device()));

        switch (grad.scalar_type()) {
            case at::ScalarType::Half:
                RecoverSequenceTrain::backward2D<__half>(
                        (const __half*)grad.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (__half*)dInp.data_ptr(),
                        batch_size,
                        seq_len,
                        hidden_size,
                        at::cuda::getCurrentCUDAStream());
                break;
            case at::ScalarType::Float:
                RecoverSequenceTrain::backward2D<float>(
                        (const float*)grad.data_ptr(),
                        (const int64_t*)length.data_ptr(),
                        (float*)dInp.data_ptr(),
                        batch_size,
                        seq_len,
                        hidden_size,
                        at::cuda::getCurrentCUDAStream());
                break;
            default:
                errMsg("invalid dtype");
        }

        return {dInp,
                torch::Tensor(),
                torch::Tensor()};
    }
};


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput2D_train(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len) {
    return RecoverSequenceAutograd2D::apply(
            inp,
            length,
            seq_len);
}

