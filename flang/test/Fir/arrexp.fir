// RUN: tco %s | FileCheck %s

func @f1(%a : !fir.ref<!fir.array<?x?xf32>>, %n : index, %m : index, %o : index, %p : index, %f : f32) {
  %c1 = constant 1 : index
  %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
  %vIn = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %r = fir.do_loop %j = %p to %m step %c1 iter_args(%v1 = %vIn) -> !fir.array<?x?xf32> {
    %r = fir.do_loop %i = %o to %n step %c1 iter_args(%v = %v1) -> !fir.array<?x?xf32> {
      %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
      fir.result %r : !fir.array<?x?xf32>
    }
    fir.result %r : !fir.array<?x?xf32>
  }
  fir.array_store %r to %a : !fir.ref<!fir.array<?x?xf32>>
  return
}
