// Sigmoid and EW product done on FP16
// 0.275 ms

// func.func @matmul(%lhs: tensor<?x14336xf16>, %rhs: tensor<?x14336xf16>, %cst_a : tensor<f32>, %cst_b : tensor<f32>, %cst_c : tensor<f32>) -> tensor<?x14336xf8E4M3FNUZ> {
//   %c0 = arith.constant 0 : index
//   %c256 = arith.constant 256 : index
//   %m = tensor.dim %lhs, %c0 : tensor<?x14336xf16>
//   %m_outer = arith.divsi %m, %c256 : index
//   %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, 14336] : tensor<?x14336xf16> into tensor<?x256x14336xf16>
  
//   %62 = tensor.empty(%m_outer) : tensor<?x256x14336xf8E4M3FNUZ>
//   %cst_0 = arith.constant 1.000000e+00 : f32
//   %cst_1 = arith.constant -2.400000e+02 : f32
//   %cst_2 = arith.constant 2.400000e+02 : f32

//   %rhs_expanded = tensor.expand_shape %rhs [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x14336xf16> into tensor<?x256x14336xf16>
//   %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%lhs_expanded, %cst_a, %rhs_expanded, %cst_b, %cst_c : tensor<?x256x14336xf16>, tensor<f32>, tensor<?x256x14336xf16>, tensor<f32>, tensor<f32>) outs(%62 : tensor<?x256x14336xf8E4M3FNUZ>) {
//   ^bb0(%in_f16: f16, %in_3: f32, %in_4_f16: f16, %in_5: f32, %in_6: f32, %out: f8E4M3FNUZ):
//     // Cast to f32
//     %in = arith.extf %in_f16 : f16 to f32
//     %in_4 = arith.extf %in_4_f16 : f16 to f32
//     // lhs, rhs scaling
//     %64 = arith.mulf %in, %in_3 : f32
//     %65 = arith.mulf %in_4, %in_5 : f32
//     // sigmoid
//     %66 = arith.negf %64 : f32
//     %67 = math.exp %66 : f32
//     %68 = arith.addf %67, %cst_0 : f32
//     %69 = arith.divf %cst_0, %68 : f32
//     //Element wise product
//     %70 = arith.mulf %69, %64 : f32
//     %71 = arith.mulf %70, %65 : f32
//     %72 = arith.divf %71, %in_6 : f32
//     //clamping & cast
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

// No sigmoid. Element wise product only 
func.func @matmul(%lhs: tensor<?x14336xf16>, %rhs: tensor<?x14336xf16>, %cst_a : tensor<f32>, %cst_b : tensor<f32>, %cst_c : tensor<f32>) -> tensor<?x14336xf8E4M3FNUZ> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x14336xf16>
  %m_outer = arith.divsi %m, %c256 : index
  %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, 14336] : tensor<?x14336xf16> into tensor<?x256x14336xf16>
  
  %62 = tensor.empty(%m_outer) : tensor<?x256x14336xf8E4M3FNUZ>
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant -2.400000e+02 : f32
  %cst_2 = arith.constant 2.400000e+02 : f32

  %rhs_expanded = tensor.expand_shape %rhs [[0, 1], [2]] output_shape [%m_outer, 256, 4096] : tensor<?x14336xf16> into tensor<?x256x14336xf16>
  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%lhs_expanded, %cst_a, %rhs_expanded, %cst_b, %cst_c : tensor<?x256x14336xf16>, tensor<f32>, tensor<?x256x14336xf16>, tensor<f32>, tensor<f32>) outs(%62 : tensor<?x256x14336xf8E4M3FNUZ>) {
  ^bb0(%in_f16: f16, %in_3: f32, %in_4_f16: f16, %in_5: f32, %in_6: f32, %out: f8E4M3FNUZ):
    // Cast to f32
    %sigmoid = arith.extf %in_f16 : f16 to f32
    %in_4 = arith.extf %in_4_f16 : f16 to f32
    // lhs, rhs scaling
    %65 = arith.mulf %in_4, %in_5 : f32

    //Element wise product
    %71 = arith.mulf %sigmoid, %65 : f32
    %72 = arith.divf %71, %in_6 : f32
    // %72 = arith.mulf %71, %in_6 : f32
    //clamping & cast
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

// func.func @matmul_dynamic(%lhs: tensor<?x14336xf16>, %rhs: tensor<?x14336xf16>, %cst_a : tensor<f32>, %cst_b : tensor<f32>, %cst_c : tensor<f32>) -> tensor<?x14336xf8E4M3FNUZ> {
//   %c0 = arith.constant 0 : index
//   %c256 = arith.constant 256 : index
//   %m = tensor.dim %lhs, %c0 : tensor<?x14336xf16>
//   %m_outer = arith.divsi %m, %c256 : index
  
//   %62 = tensor.empty(%m) : tensor<?x14336xf8E4M3FNUZ>
//   %cst_0 = arith.constant 1.000000e+00 : f32
//   %cst_1 = arith.constant -2.400000e+02 : f32
//   %cst_2 = arith.constant 2.400000e+02 : f32
//   %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [ "parallel", "parallel"]} ins(%lhs, %cst_a, %rhs, %cst_b, %cst_c : tensor<?x14336xf16>, tensor<f32>, tensor<?x14336xf16>, tensor<f32>, tensor<f32>) outs(%62 : tensor<?x14336xf8E4M3FNUZ>) {
//   ^bb0(%in_f16: f16, %in_3: f32, %in_4_f16: f16, %in_5: f32, %in_6: f32, %out: f8E4M3FNUZ):
//     %in = arith.extf %in_f16 : f16 to f32
//     %in_4 = arith.extf %in_4_f16 : f16 to f32
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
//   } -> tensor<?x14336xf8E4M3FNUZ>

  
//   return %63: tensor<?x14336xf8E4M3FNUZ>
// }