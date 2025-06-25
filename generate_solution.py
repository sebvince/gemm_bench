import numpy as np
import sys

def generate_calls_file(m: int, n: int, k: int, output_file: str):  
    content = f"""builtin.module @calls attributes {{  
    
}} {{  
  
func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view  
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)  
  
func.func private @module.matmul(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view  
  
func.func @matmul() attributes {{  
  iree.reflection = {{description = "Matmul shape (MxNxK): {m}x{n}x{k}"}}  
}} {{  
  %device_index = arith.constant 0 : index  
  %device = hal.devices.get %device_index : !hal.device  
  %lhs_dim0 = arith.constant {m} : i64  
  %lhs_dim1 = arith.constant {k} : i64  
  %lhs_element_type = hal.element_type<f8E4M3FNUZ> : i32  
  %lhs_seed = arith.constant 5 : i32  
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view  
  %rhs_dim0 = arith.constant {n} : i64  
  %rhs_dim1 = arith.constant {k} : i64  
  %rhs_element_type = hal.element_type<f8E4M3FNUZ> : i32  
  %rhs_seed = arith.constant 6 : i32  
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view  
  %acc = util.null : !hal.buffer_view  
  %result = call @module.matmul(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view  
  %m = arith.constant {m} : i64  
  %k = arith.constant {k} : i64  
  %n = arith.constant {n} : i64  
  %transpose_rhs = arith.constant 1 : i32  
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()  
  return  
}}  
  
}}"""  
    with open(output_file, "w") as f:  
        f.write(content)  


def generate_matmul_file(m: int, n: int, k: int, output_file: str):  
    content = f"""func.func @matmul(%lhs: tensor<?x{k}xf8E4M3FNUZ>, %rhs: tensor<{n}x{k}xf8E4M3FNUZ>) -> tensor<?x{n}xf32> {{
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x{k}xf8E4M3FNUZ>
  %m_outer = arith.divsi %m, %c256 : index
  %lhs_expanded = tensor.expand_shape %lhs [[0, 1], [2]] output_shape [%m_outer, 256, {k}] : tensor<?x{k}xf8E4M3FNUZ> into tensor<?x256x{k}xf8E4M3FNUZ>
  %init_acc = tensor.empty(%m_outer) : tensor<?x256x{n}xf32>
  %c0_acc_type = arith.constant 0.0: f32 
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x256x{n}xf32>) -> tensor<?x256x{n}xf32>
  %result_expanded = linalg.generic {{
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ], iterator_types = [
      "parallel", "parallel", "parallel", "reduction"
    ]
  }} ins(%lhs_expanded, %rhs : tensor<?x256x{k}xf8E4M3FNUZ>, tensor<{n}x{k}xf8E4M3FNUZ>)
    outs(%acc : tensor<?x256x{n}xf32>)
  {{
  ^bb0(%lhs_val: f8E4M3FNUZ, %rhs_val: f8E4M3FNUZ, %out: f32):
    %56 = arith.extf %lhs_val : f8E4M3FNUZ to f32
    %57 = arith.extf %rhs_val : f8E4M3FNUZ to f32
    %58 = arith.mulf %56, %57 : f32
    %59 = arith.addf %out, %58 : f32
    linalg.yield %59 : f32
  }} -> tensor<?x256x{n}xf32>
  %result = tensor.collapse_shape %result_expanded [[0, 1], [2]] : tensor<?x256x{n}xf32> into tensor<?x{n}xf32>
  return %result: tensor<?x{n}xf32>
}}"""
    with open(output_file, "w") as f:  
        f.write(content)  

if __name__ == "__main__":  
    if len(sys.argv) != 4:  
        print(f"Usage: {sys.argv[0]} <m> <n> <k>")  
        sys.exit(1)  
  
    m = int(sys.argv[1])  
    n = int(sys.argv[2])  
    k = int(sys.argv[3])  

    # generate mlir files
    generate_calls_file(m,n,k, "calls.mlir")  
    generate_matmul_file(m,n,k, "matmul.mlir")  

    # generate random data

    l = np.random.randint(low=0, high=255, size=(m, k), dtype=np.uint8)
    r = np.random.randint(low=0, high=255, size=(k, n), dtype=np.uint8)

    l.tofile("lhs.bin")
    r.tofile("rhs.bin")
