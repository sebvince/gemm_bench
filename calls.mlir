builtin.module @calls attributes {
  
} {

func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)

func.func private @module.matmul(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view

func.func @matmul() attributes {
  iree.reflection = {description = "Matmul shape (MxKxN): 2048x4096x4096"}
} {
  %device_index = arith.constant 0 : index
  %device = hal.devices.get %device_index : !hal.device
  // %lhs_dim0 = arith.constant 1280 : i64
  %lhs_dim0 = arith.constant 4096 : i64
  %lhs_dim1 = arith.constant 4096 : i64
  %lhs_element_type = hal.element_type<f8E4M3FNUZ> : i32
  %lhs_seed = arith.constant 5 : i32
  %lhs = call @matmul_test.generate_random_matrix(%device, %lhs_dim0, %lhs_dim1, %lhs_element_type, %lhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %rhs_dim0 = arith.constant 4096 : i64
  %rhs_dim1 = arith.constant 4096 : i64
  %rhs_element_type = hal.element_type<f8E4M3FNUZ> : i32
  %rhs_seed = arith.constant 6 : i32
  %rhs = call @matmul_test.generate_random_matrix(%device, %rhs_dim0, %rhs_dim1, %rhs_element_type, %rhs_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view
  %acc = util.null : !hal.buffer_view
  %result = call @module.matmul(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
  %m = arith.constant 4096 : i64
  %k = arith.constant 4096 : i64
  %n = arith.constant 4096 : i64
  %transpose_rhs = arith.constant 1 : i32
  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
  return
}

}
