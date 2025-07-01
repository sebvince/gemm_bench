// //Reference version . Matmul + sigmoid & element wise product
// func.func @matmul(%lhs: tensor<?x4096xf8E4M3FNUZ>, %rhs: tensor<14336x4096xf8E4M3FNUZ>, %prev : tensor<?x14336xf32>, %cst_a : tensor<f32>, %cst_b : tensor<f32>, %cst_c : tensor<f32>) -> tensor<?x14336xf8E4M3FNUZ> {
//   %c0 = arith.constant 0 : index
//   %c256 = arith.constant 256 : index
//   %m = tensor.dim %lhs, %c0 : tensor<?x4096xf8E4M3FNUZ>
//   %m_outer = arith.divsi %m, %c256 : index
//   %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x4096xf8E4M3FNUZ> into tensor<?x256x4096xf8E4M3FNUZ>
//   %init_acc = tensor.empty(%m_outer) : tensor<?x256x14336xf32>
//   %c0_acc_type = arith.constant 0.0: f32 
//   %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x256x14336xf32>) -> tensor<?x256x14336xf32>
//   %result_expanded = linalg.generic {
//     indexing_maps = [
//       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
//       affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
//       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//     ], iterator_types = [
//       "parallel", "parallel", "parallel", "reduction"
//     ]
//   } ins(%lhs_expanded, %rhs : tensor<?x256x4096xf8E4M3FNUZ>, tensor<14336x4096xf8E4M3FNUZ>)
//     outs(%acc : tensor<?x256x14336xf32>)
//   {
//   ^bb0(%lhs_val: f8E4M3FNUZ, %rhs_val: f8E4M3FNUZ, %out: f32):
//     %56 = arith.extf %lhs_val : f8E4M3FNUZ to f32
//     %57 = arith.extf %rhs_val : f8E4M3FNUZ to f32
//     %58 = arith.mulf %56, %57 : f32
//     %59 = arith.addf %out, %58 : f32
//     linalg.yield %59 : f32
//   } -> tensor<?x256x14336xf32>

//   %62 = tensor.empty(%m_outer) : tensor<?x256x14336xf8E4M3FNUZ>
//   %cst_0 = arith.constant 1.000000e+00 : f32
//   %cst_1 = arith.constant -2.400000e+02 : f32
//   %cst_2 = arith.constant 2.400000e+02 : f32

//   %prev_expanded = tensor.expand_shape %prev [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x14336xf32> into tensor<?x256x14336xf32>
//   %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%result_expanded, %cst_a, %prev_expanded, %cst_b, %cst_c : tensor<?x256x14336xf32>, tensor<f32>, tensor<?x256x14336xf32>, tensor<f32>, tensor<f32>) outs(%62 : tensor<?x256x14336xf8E4M3FNUZ>) {
//   ^bb0(%in: f32, %in_3: f32, %in_4: f32, %in_5: f32, %in_6: f32, %out: f8E4M3FNUZ):
//     %64 = arith.mulf %in, %in_3 : f32
//     %65 = arith.mulf %in_4, %in_5 : f32
//     %66 = arith.negf %64 : f32
//     %67 = math.exp %66 : f32
//     %68 = arith.addf %67, %cst_0 : f32
//     %69 = arith.divf %cst_0, %68 : f32
//     %70 = arith.mulf %69, %64 : f32
//     %71 = arith.mulf %70, %65 : f32
//     %72 = arith.divf %71, %in_6 : f32
//     %73 = arith.cmpf ult, %72, %cst_1 : f32
//     %74 = arith.select %73, %cst_1, %72 : f32
//     %75 = arith.cmpf ugt, %74, %cst_2 : f32
//     %76 = arith.select %75, %cst_2, %74 : f32
//     %77 = arith.truncf %76 : f32 to f8E4M3FNUZ
//     linalg.yield %77 : f8E4M3FNUZ
//   } -> tensor<?x256x14336xf8E4M3FNUZ>

//   %result = tensor.collapse_shape %63 [[0, 1], [2]] : tensor<?x256x14336xf8E4M3FNUZ> into tensor<?x14336xf8E4M3FNUZ>
//   return %result: tensor<?x14336xf8E4M3FNUZ>
// }

// //Matmul + sigmoid  + cast to f16
// func.func @matmul(%lhs: tensor<?x4096xf8E4M3FNUZ>, %rhs: tensor<14336x4096xf8E4M3FNUZ>, %cst_a : tensor<f32>, %cst_b : tensor<f32>, %cst_c : tensor<f32>) -> tensor<?x14336xf16> {
//   %c0 = arith.constant 0 : index
//   %c256 = arith.constant 256 : index
//   %m = tensor.dim %lhs, %c0 : tensor<?x4096xf8E4M3FNUZ>
//   %m_outer = arith.divsi %m, %c256 : index
//   %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x4096xf8E4M3FNUZ> into tensor<?x256x4096xf8E4M3FNUZ>
//   %init_acc = tensor.empty(%m_outer) : tensor<?x256x14336xf32>
//   %c0_acc_type = arith.constant 0.0: f32 
//   %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x256x14336xf32>) -> tensor<?x256x14336xf32>
//   %result_expanded = linalg.generic {
//     indexing_maps = [
//       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
//       affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
//       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//     ], iterator_types = [
//       "parallel", "parallel", "parallel", "reduction"
//     ]
//   } ins(%lhs_expanded, %rhs : tensor<?x256x4096xf8E4M3FNUZ>, tensor<14336x4096xf8E4M3FNUZ>)
//     outs(%acc : tensor<?x256x14336xf32>)
//   {
//   ^bb0(%lhs_val: f8E4M3FNUZ, %rhs_val: f8E4M3FNUZ, %out: f32):
//     %56 = arith.extf %lhs_val : f8E4M3FNUZ to f32
//     %57 = arith.extf %rhs_val : f8E4M3FNUZ to f32
//     %58 = arith.mulf %56, %57 : f32
//     %59 = arith.addf %out, %58 : f32
//     linalg.yield %59 : f32
//   } -> tensor<?x256x14336xf32>

//   %62 = tensor.empty(%m_outer) : tensor<?x256x14336xf16>
//   %cst_0 = arith.constant 1.000000e+00 : f32
//   %cst_1 = arith.constant -2.400000e+02 : f32
//   %cst_2 = arith.constant 2.400000e+02 : f32

//   %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%result_expanded, %cst_a, %cst_b, %cst_c : tensor<?x256x14336xf32>, tensor<f32>, tensor<f32>, tensor<f32>) outs(%62 : tensor<?x256x14336xf16>) {
//   ^bb0(%in: f32, %in_3: f32, %in_5: f32, %in_6: f32, %out: f16):
//     %64 = arith.mulf %in, %in_3 : f32
    
//     %66 = arith.negf %64 : f32
//     %67 = math.exp %66 : f32
//     %68 = arith.addf %67, %cst_0 : f32
//     %69 = arith.divf %cst_0, %68 : f32
//     %70 = arith.mulf %69, %64 : f32

//     %77 = arith.truncf %70 : f32 to f16
//     linalg.yield %77 : f16
//   } -> tensor<?x256x14336xf16>

//   %result = tensor.collapse_shape %63 [[0, 1], [2]] : tensor<?x256x14336xf16> into tensor<?x14336xf16>
//   return %result: tensor<?x14336xf16>
// }

//Matmul + element wise product
func.func @matmul(%lhs: tensor<?x4096xf8E4M3FNUZ>, %rhs: tensor<14336x4096xf8E4M3FNUZ>, %prev : tensor<?x14336xf16>, %cst_a : tensor<f32>, %cst_b : tensor<f32>, %cst_c : tensor<f32>) -> tensor<?x14336xf8E4M3FNUZ> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x4096xf8E4M3FNUZ>
  %m_outer = arith.divsi %m, %c256 : index
  %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x4096xf8E4M3FNUZ> into tensor<?x256x4096xf8E4M3FNUZ>
  %init_acc = tensor.empty(%m_outer) : tensor<?x256x14336xf32>
  %c0_acc_type = arith.constant 0.0: f32 
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x256x14336xf32>) -> tensor<?x256x14336xf32>
  %result_expanded = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ], iterator_types = [
      "parallel", "parallel", "parallel", "reduction"
    ]
  } ins(%lhs_expanded, %rhs : tensor<?x256x4096xf8E4M3FNUZ>, tensor<14336x4096xf8E4M3FNUZ>)
    outs(%acc : tensor<?x256x14336xf32>)
  {
  ^bb0(%lhs_val: f8E4M3FNUZ, %rhs_val: f8E4M3FNUZ, %out: f32):
    %56 = arith.extf %lhs_val : f8E4M3FNUZ to f32
    %57 = arith.extf %rhs_val : f8E4M3FNUZ to f32
    %58 = arith.mulf %56, %57 : f32
    %59 = arith.addf %out, %58 : f32
    linalg.yield %59 : f32
  } -> tensor<?x256x14336xf32>

  %62 = tensor.empty(%m_outer) : tensor<?x256x14336xf8E4M3FNUZ>
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant -2.400000e+02 : f32
  %cst_2 = arith.constant 2.400000e+02 : f32

  %prev_expanded = tensor.expand_shape %prev [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x14336xf16> into tensor<?x256x14336xf16>
  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%result_expanded, %cst_a, %prev_expanded, %cst_b, %cst_c : tensor<?x256x14336xf32>, tensor<f32>, tensor<?x256x14336xf16>, tensor<f32>, tensor<f32>) outs(%62 : tensor<?x256x14336xf8E4M3FNUZ>) {
  ^bb0(%in: f32, %in_3: f32, %in_4_f16: f16, %in_5: f32, %in_6: f32, %out: f8E4M3FNUZ):
    //cast
    %in_4 = arith.extf %in_4_f16 : f16 to f32
    //scaling
    %64 = arith.mulf %in, %in_3 : f32
    //product    
    %71 = arith.mulf %in_4, %64 : f32
    //scaling
    %72 = arith.divf %71, %cst_2 : f32 
    //clamping
    %73 = arith.cmpf ult, %72, %cst_1 : f32
    %74 = arith.select %73, %cst_1, %72 : f32
    %75 = arith.cmpf ugt, %74, %cst_2 : f32
    %76 = arith.select %75, %cst_2, %74 : f32
    %77 = arith.truncf %76 : f32 to f8E4M3FNUZ
    linalg.yield %77 : f8E4M3FNUZ
  } -> tensor<?x256x14336xf8E4M3FNUZ>

  %result = tensor.collapse_shape %63 [[0, 1], [2]] : tensor<?x256x14336xf8E4M3FNUZ> into tensor<?x14336xf8E4M3FNUZ>
  return %result: tensor<?x14336xf8E4M3FNUZ>
}