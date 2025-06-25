func.func @matmul(%lhs: tensor<?x14336xf8E4M3FNUZ>, %rhs: tensor<4096x14336xf8E4M3FNUZ>) -> tensor<?x4096xf32> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x14336xf8E4M3FNUZ>
  %m_outer = arith.divsi %m, %c256 : index
  %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, 14336] : tensor<?x14336xf8E4M3FNUZ> into tensor<?x256x14336xf8E4M3FNUZ>
  %init_acc = tensor.empty(%m_outer) : tensor<?x256x4096xf32>
  %c0_acc_type = arith.constant 0.0: f32 
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x256x4096xf32>) -> tensor<?x256x4096xf32>
  %result_expanded = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ], iterator_types = [
      "parallel", "parallel", "parallel", "reduction"
    ]
  } ins(%lhs_expanded, %rhs : tensor<?x256x14336xf8E4M3FNUZ>, tensor<4096x14336xf8E4M3FNUZ>)
    outs(%acc : tensor<?x256x4096xf32>)
  {
  ^bb0(%lhs_val: f8E4M3FNUZ, %rhs_val: f8E4M3FNUZ, %out: f32):
    %56 = arith.extf %lhs_val : f8E4M3FNUZ to f32
    %57 = arith.extf %rhs_val : f8E4M3FNUZ to f32
    %58 = arith.mulf %56, %57 : f32
    %59 = arith.addf %out, %58 : f32
    linalg.yield %59 : f32
  } -> tensor<?x256x4096xf32>
  %result = tensor.collapse_shape %result_expanded [[0, 1], [2]] : tensor<?x256x4096xf32> into tensor<?x4096xf32>
  return %result: tensor<?x4096xf32>
}
