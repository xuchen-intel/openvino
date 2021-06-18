// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MKLDNN_EXTENSION_NODE
# warning "MKLDNN_EXTENSION_NODE is not defined"
# define MKLDNN_EXTENSION_NODE(__prim, __type)
#endif

MKLDNN_EXTENSION_NODE(CTCLossImpl, CTCLoss);
MKLDNN_EXTENSION_NODE(MathImpl, Abs);
MKLDNN_EXTENSION_NODE(MathImpl, Acos);
MKLDNN_EXTENSION_NODE(MathImpl, Acosh);
MKLDNN_EXTENSION_NODE(MathImpl, Asin);
MKLDNN_EXTENSION_NODE(MathImpl, Asinh);
MKLDNN_EXTENSION_NODE(MathImpl, Atan);
MKLDNN_EXTENSION_NODE(MathImpl, Atanh);
MKLDNN_EXTENSION_NODE(MathImpl, Ceil);
MKLDNN_EXTENSION_NODE(MathImpl, Ceiling);
MKLDNN_EXTENSION_NODE(MathImpl, Cos);
MKLDNN_EXTENSION_NODE(MathImpl, Cosh);
MKLDNN_EXTENSION_NODE(MathImpl, Floor);
MKLDNN_EXTENSION_NODE(MathImpl, HardSigmoid);
MKLDNN_EXTENSION_NODE(MathImpl, Log);
MKLDNN_EXTENSION_NODE(MathImpl, Neg);
MKLDNN_EXTENSION_NODE(MathImpl, Reciprocal);
MKLDNN_EXTENSION_NODE(MathImpl, Selu);
MKLDNN_EXTENSION_NODE(MathImpl, Sign);
MKLDNN_EXTENSION_NODE(MathImpl, Sin);
MKLDNN_EXTENSION_NODE(MathImpl, Sinh);
MKLDNN_EXTENSION_NODE(MathImpl, Softsign);
MKLDNN_EXTENSION_NODE(MathImpl, Tan);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronTopKROIsImpl, ExperimentalDetectronTopKROIs);
MKLDNN_EXTENSION_NODE(ExtractImagePatchesImpl, ExtractImagePatches);
MKLDNN_EXTENSION_NODE(ReverseSequenceImpl, ReverseSequence);
MKLDNN_EXTENSION_NODE(DetectionOutputImpl, DetectionOutput);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronDetectionOutputImpl, ExperimentalDetectronDetectionOutput);
MKLDNN_EXTENSION_NODE(LogSoftmaxImpl, LogSoftmax);
MKLDNN_EXTENSION_NODE(ReorgYoloImpl, ReorgYolo);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronROIFeatureExtractorImpl, ExperimentalDetectronROIFeatureExtractor);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronGenerateProposalsSingleImageImpl, ExperimentalDetectronGenerateProposalsSingleImage);
MKLDNN_EXTENSION_NODE(NonMaxSuppressionImpl, NonMaxSuppressionIEInternal);
MKLDNN_EXTENSION_NODE(ShuffleChannelsImpl, ShuffleChannels);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronPriorGridGeneratorImpl, ExperimentalDetectronPriorGridGenerator);
MKLDNN_EXTENSION_NODE(GRNImpl, GRN);
MKLDNN_EXTENSION_NODE(BucketizeImpl, Bucketize);
MKLDNN_EXTENSION_NODE(CTCGreedyDecoderImpl, CTCGreedyDecoder);
MKLDNN_EXTENSION_NODE(CTCGreedyDecoderSeqLenImpl, CTCGreedyDecoderSeqLen);
MKLDNN_EXTENSION_NODE(ProposalImpl, Proposal);
MKLDNN_EXTENSION_NODE(RangeImpl, Range);
MKLDNN_EXTENSION_NODE(GatherTreeImpl, GatherTree);
MKLDNN_EXTENSION_NODE(CumSumImpl, CumSum);
