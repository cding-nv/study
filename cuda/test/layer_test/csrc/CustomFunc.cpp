#include <torch/extension.h>


torch::Tensor
AdjMatrixBatchSimpleGenerate(
        const torch::Tensor& inp,
        double alpha);


torch::Tensor
BatchingSequenceOfSequenceDataReduceInvalidInputA(
        const torch::Tensor& inp,
        int64_t seq_len);


torch::Tensor
BatchingSequenceOfSequenceDataReduceInvalidInputB(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t length_sum,
        int64_t seq_len);


std::vector<torch::Tensor>
PositionsAndTimeDiff(
        const torch::Tensor& time,
        const torch::Tensor& mask);


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len);


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput1D_train(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len);


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len);


torch::Tensor
RecoverSequenceOfSequenceDataReduceInvalidInput2D_train(
        const torch::Tensor& inp,
        const torch::Tensor& length,
        int64_t seq_len);


TORCH_LIBRARY(func, m) {
    m.def("AdjMatrixBatchSimpleGenerate",
          &AdjMatrixBatchSimpleGenerate);
    m.def("BatchingSequenceOfSequenceDataReduceInvalidInputA",
          &BatchingSequenceOfSequenceDataReduceInvalidInputA);
    m.def("BatchingSequenceOfSequenceDataReduceInvalidInputB",
          &BatchingSequenceOfSequenceDataReduceInvalidInputB);
    m.def("PositionsAndTimeDiff",
          &PositionsAndTimeDiff);
    m.def("RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer",
          &RecoverSequenceOfSequenceDataReduceInvalidInput1D_infer);
    m.def("RecoverSequenceOfSequenceDataReduceInvalidInput1D_train",
          &RecoverSequenceOfSequenceDataReduceInvalidInput1D_train);
    m.def("RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer",
          &RecoverSequenceOfSequenceDataReduceInvalidInput2D_infer);
    m.def("RecoverSequenceOfSequenceDataReduceInvalidInput2D_train",
          &RecoverSequenceOfSequenceDataReduceInvalidInput2D_train);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

