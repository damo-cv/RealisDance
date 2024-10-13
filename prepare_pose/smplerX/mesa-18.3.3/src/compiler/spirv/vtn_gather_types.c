/*
 * Copyright (C) 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* DO NOT EDIT - This file is generated automatically by the
 * vtn_gather_types_c.py script
 */

#include "vtn_private.h"

struct type_args {
    int res_idx;
    int res_type_idx;
};

static struct type_args
result_type_args_for_opcode(SpvOp opcode)
{
   switch (opcode) {
   case SpvOpUndef: return (struct type_args){ 1, 0 };
   case SpvOpString: return (struct type_args){ 0, -1 };
   case SpvOpExtInstImport: return (struct type_args){ 0, -1 };
   case SpvOpExtInst: return (struct type_args){ 1, 0 };
   case SpvOpTypeVoid: return (struct type_args){ 0, -1 };
   case SpvOpTypeBool: return (struct type_args){ 0, -1 };
   case SpvOpTypeInt: return (struct type_args){ 0, -1 };
   case SpvOpTypeFloat: return (struct type_args){ 0, -1 };
   case SpvOpTypeVector: return (struct type_args){ 0, -1 };
   case SpvOpTypeMatrix: return (struct type_args){ 0, -1 };
   case SpvOpTypeImage: return (struct type_args){ 0, -1 };
   case SpvOpTypeSampler: return (struct type_args){ 0, -1 };
   case SpvOpTypeSampledImage: return (struct type_args){ 0, -1 };
   case SpvOpTypeArray: return (struct type_args){ 0, -1 };
   case SpvOpTypeRuntimeArray: return (struct type_args){ 0, -1 };
   case SpvOpTypeStruct: return (struct type_args){ 0, -1 };
   case SpvOpTypeOpaque: return (struct type_args){ 0, -1 };
   case SpvOpTypePointer: return (struct type_args){ 0, -1 };
   case SpvOpTypeFunction: return (struct type_args){ 0, -1 };
   case SpvOpTypeEvent: return (struct type_args){ 0, -1 };
   case SpvOpTypeDeviceEvent: return (struct type_args){ 0, -1 };
   case SpvOpTypeReserveId: return (struct type_args){ 0, -1 };
   case SpvOpTypeQueue: return (struct type_args){ 0, -1 };
   case SpvOpTypePipe: return (struct type_args){ 0, -1 };
   case SpvOpConstantTrue: return (struct type_args){ 1, 0 };
   case SpvOpConstantFalse: return (struct type_args){ 1, 0 };
   case SpvOpConstant: return (struct type_args){ 1, 0 };
   case SpvOpConstantComposite: return (struct type_args){ 1, 0 };
   case SpvOpConstantSampler: return (struct type_args){ 1, 0 };
   case SpvOpConstantNull: return (struct type_args){ 1, 0 };
   case SpvOpSpecConstantTrue: return (struct type_args){ 1, 0 };
   case SpvOpSpecConstantFalse: return (struct type_args){ 1, 0 };
   case SpvOpSpecConstant: return (struct type_args){ 1, 0 };
   case SpvOpSpecConstantComposite: return (struct type_args){ 1, 0 };
   case SpvOpSpecConstantOp: return (struct type_args){ 1, 0 };
   case SpvOpFunction: return (struct type_args){ 1, 0 };
   case SpvOpFunctionParameter: return (struct type_args){ 1, 0 };
   case SpvOpFunctionCall: return (struct type_args){ 1, 0 };
   case SpvOpVariable: return (struct type_args){ 1, 0 };
   case SpvOpImageTexelPointer: return (struct type_args){ 1, 0 };
   case SpvOpLoad: return (struct type_args){ 1, 0 };
   case SpvOpAccessChain: return (struct type_args){ 1, 0 };
   case SpvOpInBoundsAccessChain: return (struct type_args){ 1, 0 };
   case SpvOpPtrAccessChain: return (struct type_args){ 1, 0 };
   case SpvOpArrayLength: return (struct type_args){ 1, 0 };
   case SpvOpGenericPtrMemSemantics: return (struct type_args){ 1, 0 };
   case SpvOpInBoundsPtrAccessChain: return (struct type_args){ 1, 0 };
   case SpvOpDecorationGroup: return (struct type_args){ 0, -1 };
   case SpvOpVectorExtractDynamic: return (struct type_args){ 1, 0 };
   case SpvOpVectorInsertDynamic: return (struct type_args){ 1, 0 };
   case SpvOpVectorShuffle: return (struct type_args){ 1, 0 };
   case SpvOpCompositeConstruct: return (struct type_args){ 1, 0 };
   case SpvOpCompositeExtract: return (struct type_args){ 1, 0 };
   case SpvOpCompositeInsert: return (struct type_args){ 1, 0 };
   case SpvOpCopyObject: return (struct type_args){ 1, 0 };
   case SpvOpTranspose: return (struct type_args){ 1, 0 };
   case SpvOpSampledImage: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleDrefImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleDrefExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleProjImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleProjExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleProjDrefImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleProjDrefExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageFetch: return (struct type_args){ 1, 0 };
   case SpvOpImageGather: return (struct type_args){ 1, 0 };
   case SpvOpImageDrefGather: return (struct type_args){ 1, 0 };
   case SpvOpImageRead: return (struct type_args){ 1, 0 };
   case SpvOpImage: return (struct type_args){ 1, 0 };
   case SpvOpImageQueryFormat: return (struct type_args){ 1, 0 };
   case SpvOpImageQueryOrder: return (struct type_args){ 1, 0 };
   case SpvOpImageQuerySizeLod: return (struct type_args){ 1, 0 };
   case SpvOpImageQuerySize: return (struct type_args){ 1, 0 };
   case SpvOpImageQueryLod: return (struct type_args){ 1, 0 };
   case SpvOpImageQueryLevels: return (struct type_args){ 1, 0 };
   case SpvOpImageQuerySamples: return (struct type_args){ 1, 0 };
   case SpvOpConvertFToU: return (struct type_args){ 1, 0 };
   case SpvOpConvertFToS: return (struct type_args){ 1, 0 };
   case SpvOpConvertSToF: return (struct type_args){ 1, 0 };
   case SpvOpConvertUToF: return (struct type_args){ 1, 0 };
   case SpvOpUConvert: return (struct type_args){ 1, 0 };
   case SpvOpSConvert: return (struct type_args){ 1, 0 };
   case SpvOpFConvert: return (struct type_args){ 1, 0 };
   case SpvOpQuantizeToF16: return (struct type_args){ 1, 0 };
   case SpvOpConvertPtrToU: return (struct type_args){ 1, 0 };
   case SpvOpSatConvertSToU: return (struct type_args){ 1, 0 };
   case SpvOpSatConvertUToS: return (struct type_args){ 1, 0 };
   case SpvOpConvertUToPtr: return (struct type_args){ 1, 0 };
   case SpvOpPtrCastToGeneric: return (struct type_args){ 1, 0 };
   case SpvOpGenericCastToPtr: return (struct type_args){ 1, 0 };
   case SpvOpGenericCastToPtrExplicit: return (struct type_args){ 1, 0 };
   case SpvOpBitcast: return (struct type_args){ 1, 0 };
   case SpvOpSNegate: return (struct type_args){ 1, 0 };
   case SpvOpFNegate: return (struct type_args){ 1, 0 };
   case SpvOpIAdd: return (struct type_args){ 1, 0 };
   case SpvOpFAdd: return (struct type_args){ 1, 0 };
   case SpvOpISub: return (struct type_args){ 1, 0 };
   case SpvOpFSub: return (struct type_args){ 1, 0 };
   case SpvOpIMul: return (struct type_args){ 1, 0 };
   case SpvOpFMul: return (struct type_args){ 1, 0 };
   case SpvOpUDiv: return (struct type_args){ 1, 0 };
   case SpvOpSDiv: return (struct type_args){ 1, 0 };
   case SpvOpFDiv: return (struct type_args){ 1, 0 };
   case SpvOpUMod: return (struct type_args){ 1, 0 };
   case SpvOpSRem: return (struct type_args){ 1, 0 };
   case SpvOpSMod: return (struct type_args){ 1, 0 };
   case SpvOpFRem: return (struct type_args){ 1, 0 };
   case SpvOpFMod: return (struct type_args){ 1, 0 };
   case SpvOpVectorTimesScalar: return (struct type_args){ 1, 0 };
   case SpvOpMatrixTimesScalar: return (struct type_args){ 1, 0 };
   case SpvOpVectorTimesMatrix: return (struct type_args){ 1, 0 };
   case SpvOpMatrixTimesVector: return (struct type_args){ 1, 0 };
   case SpvOpMatrixTimesMatrix: return (struct type_args){ 1, 0 };
   case SpvOpOuterProduct: return (struct type_args){ 1, 0 };
   case SpvOpDot: return (struct type_args){ 1, 0 };
   case SpvOpIAddCarry: return (struct type_args){ 1, 0 };
   case SpvOpISubBorrow: return (struct type_args){ 1, 0 };
   case SpvOpUMulExtended: return (struct type_args){ 1, 0 };
   case SpvOpSMulExtended: return (struct type_args){ 1, 0 };
   case SpvOpAny: return (struct type_args){ 1, 0 };
   case SpvOpAll: return (struct type_args){ 1, 0 };
   case SpvOpIsNan: return (struct type_args){ 1, 0 };
   case SpvOpIsInf: return (struct type_args){ 1, 0 };
   case SpvOpIsFinite: return (struct type_args){ 1, 0 };
   case SpvOpIsNormal: return (struct type_args){ 1, 0 };
   case SpvOpSignBitSet: return (struct type_args){ 1, 0 };
   case SpvOpLessOrGreater: return (struct type_args){ 1, 0 };
   case SpvOpOrdered: return (struct type_args){ 1, 0 };
   case SpvOpUnordered: return (struct type_args){ 1, 0 };
   case SpvOpLogicalEqual: return (struct type_args){ 1, 0 };
   case SpvOpLogicalNotEqual: return (struct type_args){ 1, 0 };
   case SpvOpLogicalOr: return (struct type_args){ 1, 0 };
   case SpvOpLogicalAnd: return (struct type_args){ 1, 0 };
   case SpvOpLogicalNot: return (struct type_args){ 1, 0 };
   case SpvOpSelect: return (struct type_args){ 1, 0 };
   case SpvOpIEqual: return (struct type_args){ 1, 0 };
   case SpvOpINotEqual: return (struct type_args){ 1, 0 };
   case SpvOpUGreaterThan: return (struct type_args){ 1, 0 };
   case SpvOpSGreaterThan: return (struct type_args){ 1, 0 };
   case SpvOpUGreaterThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpSGreaterThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpULessThan: return (struct type_args){ 1, 0 };
   case SpvOpSLessThan: return (struct type_args){ 1, 0 };
   case SpvOpULessThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpSLessThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpFOrdEqual: return (struct type_args){ 1, 0 };
   case SpvOpFUnordEqual: return (struct type_args){ 1, 0 };
   case SpvOpFOrdNotEqual: return (struct type_args){ 1, 0 };
   case SpvOpFUnordNotEqual: return (struct type_args){ 1, 0 };
   case SpvOpFOrdLessThan: return (struct type_args){ 1, 0 };
   case SpvOpFUnordLessThan: return (struct type_args){ 1, 0 };
   case SpvOpFOrdGreaterThan: return (struct type_args){ 1, 0 };
   case SpvOpFUnordGreaterThan: return (struct type_args){ 1, 0 };
   case SpvOpFOrdLessThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpFUnordLessThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpFOrdGreaterThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpFUnordGreaterThanEqual: return (struct type_args){ 1, 0 };
   case SpvOpShiftRightLogical: return (struct type_args){ 1, 0 };
   case SpvOpShiftRightArithmetic: return (struct type_args){ 1, 0 };
   case SpvOpShiftLeftLogical: return (struct type_args){ 1, 0 };
   case SpvOpBitwiseOr: return (struct type_args){ 1, 0 };
   case SpvOpBitwiseXor: return (struct type_args){ 1, 0 };
   case SpvOpBitwiseAnd: return (struct type_args){ 1, 0 };
   case SpvOpNot: return (struct type_args){ 1, 0 };
   case SpvOpBitFieldInsert: return (struct type_args){ 1, 0 };
   case SpvOpBitFieldSExtract: return (struct type_args){ 1, 0 };
   case SpvOpBitFieldUExtract: return (struct type_args){ 1, 0 };
   case SpvOpBitReverse: return (struct type_args){ 1, 0 };
   case SpvOpBitCount: return (struct type_args){ 1, 0 };
   case SpvOpDPdx: return (struct type_args){ 1, 0 };
   case SpvOpDPdy: return (struct type_args){ 1, 0 };
   case SpvOpFwidth: return (struct type_args){ 1, 0 };
   case SpvOpDPdxFine: return (struct type_args){ 1, 0 };
   case SpvOpDPdyFine: return (struct type_args){ 1, 0 };
   case SpvOpFwidthFine: return (struct type_args){ 1, 0 };
   case SpvOpDPdxCoarse: return (struct type_args){ 1, 0 };
   case SpvOpDPdyCoarse: return (struct type_args){ 1, 0 };
   case SpvOpFwidthCoarse: return (struct type_args){ 1, 0 };
   case SpvOpAtomicLoad: return (struct type_args){ 1, 0 };
   case SpvOpAtomicExchange: return (struct type_args){ 1, 0 };
   case SpvOpAtomicCompareExchange: return (struct type_args){ 1, 0 };
   case SpvOpAtomicCompareExchangeWeak: return (struct type_args){ 1, 0 };
   case SpvOpAtomicIIncrement: return (struct type_args){ 1, 0 };
   case SpvOpAtomicIDecrement: return (struct type_args){ 1, 0 };
   case SpvOpAtomicIAdd: return (struct type_args){ 1, 0 };
   case SpvOpAtomicISub: return (struct type_args){ 1, 0 };
   case SpvOpAtomicSMin: return (struct type_args){ 1, 0 };
   case SpvOpAtomicUMin: return (struct type_args){ 1, 0 };
   case SpvOpAtomicSMax: return (struct type_args){ 1, 0 };
   case SpvOpAtomicUMax: return (struct type_args){ 1, 0 };
   case SpvOpAtomicAnd: return (struct type_args){ 1, 0 };
   case SpvOpAtomicOr: return (struct type_args){ 1, 0 };
   case SpvOpAtomicXor: return (struct type_args){ 1, 0 };
   case SpvOpPhi: return (struct type_args){ 1, 0 };
   case SpvOpLabel: return (struct type_args){ 0, -1 };
   case SpvOpGroupAsyncCopy: return (struct type_args){ 1, 0 };
   case SpvOpGroupAll: return (struct type_args){ 1, 0 };
   case SpvOpGroupAny: return (struct type_args){ 1, 0 };
   case SpvOpGroupBroadcast: return (struct type_args){ 1, 0 };
   case SpvOpGroupIAdd: return (struct type_args){ 1, 0 };
   case SpvOpGroupFAdd: return (struct type_args){ 1, 0 };
   case SpvOpGroupFMin: return (struct type_args){ 1, 0 };
   case SpvOpGroupUMin: return (struct type_args){ 1, 0 };
   case SpvOpGroupSMin: return (struct type_args){ 1, 0 };
   case SpvOpGroupFMax: return (struct type_args){ 1, 0 };
   case SpvOpGroupUMax: return (struct type_args){ 1, 0 };
   case SpvOpGroupSMax: return (struct type_args){ 1, 0 };
   case SpvOpReadPipe: return (struct type_args){ 1, 0 };
   case SpvOpWritePipe: return (struct type_args){ 1, 0 };
   case SpvOpReservedReadPipe: return (struct type_args){ 1, 0 };
   case SpvOpReservedWritePipe: return (struct type_args){ 1, 0 };
   case SpvOpReserveReadPipePackets: return (struct type_args){ 1, 0 };
   case SpvOpReserveWritePipePackets: return (struct type_args){ 1, 0 };
   case SpvOpIsValidReserveId: return (struct type_args){ 1, 0 };
   case SpvOpGetNumPipePackets: return (struct type_args){ 1, 0 };
   case SpvOpGetMaxPipePackets: return (struct type_args){ 1, 0 };
   case SpvOpGroupReserveReadPipePackets: return (struct type_args){ 1, 0 };
   case SpvOpGroupReserveWritePipePackets: return (struct type_args){ 1, 0 };
   case SpvOpEnqueueMarker: return (struct type_args){ 1, 0 };
   case SpvOpEnqueueKernel: return (struct type_args){ 1, 0 };
   case SpvOpGetKernelNDrangeSubGroupCount: return (struct type_args){ 1, 0 };
   case SpvOpGetKernelNDrangeMaxSubGroupSize: return (struct type_args){ 1, 0 };
   case SpvOpGetKernelWorkGroupSize: return (struct type_args){ 1, 0 };
   case SpvOpGetKernelPreferredWorkGroupSizeMultiple: return (struct type_args){ 1, 0 };
   case SpvOpCreateUserEvent: return (struct type_args){ 1, 0 };
   case SpvOpIsValidEvent: return (struct type_args){ 1, 0 };
   case SpvOpGetDefaultQueue: return (struct type_args){ 1, 0 };
   case SpvOpBuildNDRange: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleDrefImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleDrefExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleProjImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleProjExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleProjDrefImplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseSampleProjDrefExplicitLod: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseFetch: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseGather: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseDrefGather: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseTexelsResident: return (struct type_args){ 1, 0 };
   case SpvOpAtomicFlagTestAndSet: return (struct type_args){ 1, 0 };
   case SpvOpImageSparseRead: return (struct type_args){ 1, 0 };
   case SpvOpSizeOf: return (struct type_args){ 1, 0 };
   case SpvOpTypePipeStorage: return (struct type_args){ 0, -1 };
   case SpvOpConstantPipeStorage: return (struct type_args){ 1, 0 };
   case SpvOpCreatePipeFromPipeStorage: return (struct type_args){ 1, 0 };
   case SpvOpGetKernelLocalSizeForSubgroupCount: return (struct type_args){ 1, 0 };
   case SpvOpGetKernelMaxNumSubgroups: return (struct type_args){ 1, 0 };
   case SpvOpTypeNamedBarrier: return (struct type_args){ 0, -1 };
   case SpvOpNamedBarrierInitialize: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformElect: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformAll: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformAny: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformAllEqual: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBroadcast: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBroadcastFirst: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBallot: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformInverseBallot: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBallotBitExtract: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBallotBitCount: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBallotFindLSB: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBallotFindMSB: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformShuffle: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformShuffleXor: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformShuffleUp: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformShuffleDown: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformIAdd: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformFAdd: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformIMul: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformFMul: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformSMin: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformUMin: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformFMin: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformSMax: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformUMax: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformFMax: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBitwiseAnd: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBitwiseOr: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformBitwiseXor: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformLogicalAnd: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformLogicalOr: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformLogicalXor: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformQuadBroadcast: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformQuadSwap: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupBallotKHR: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupFirstInvocationKHR: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupAllKHR: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupAnyKHR: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupAllEqualKHR: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupReadInvocationKHR: return (struct type_args){ 1, 0 };
   case SpvOpGroupIAddNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupFAddNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupFMinNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupUMinNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupSMinNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupFMaxNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupUMaxNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpGroupSMaxNonUniformAMD: return (struct type_args){ 1, 0 };
   case SpvOpFragmentMaskFetchAMD: return (struct type_args){ 1, 0 };
   case SpvOpFragmentFetchAMD: return (struct type_args){ 1, 0 };
   case SpvOpReportIntersectionNVX: return (struct type_args){ 1, 0 };
   case SpvOpTypeAccelerationStructureNVX: return (struct type_args){ 0, -1 };
   case SpvOpSubgroupShuffleINTEL: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupShuffleDownINTEL: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupShuffleUpINTEL: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupShuffleXorINTEL: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupBlockReadINTEL: return (struct type_args){ 1, 0 };
   case SpvOpSubgroupImageBlockReadINTEL: return (struct type_args){ 1, 0 };
   case SpvOpGroupNonUniformPartitionNV: return (struct type_args){ 1, 0 };
   case SpvOpImageSampleFootprintNV: return (struct type_args){ 1, 0 };
   default: return (struct type_args){ -1, -1 };
   }
}

bool
vtn_set_instruction_result_type(struct vtn_builder *b, SpvOp opcode,
                                const uint32_t *w, unsigned count)
{
   struct type_args args = result_type_args_for_opcode(opcode);

   if (args.res_idx >= 0 && args.res_type_idx >= 0) {
      struct vtn_value *val = vtn_untyped_value(b, w[1 + args.res_idx]);
      val->type = vtn_value(b, w[1 + args.res_type_idx],
                            vtn_value_type_type)->type;
   }

   return true;
}

