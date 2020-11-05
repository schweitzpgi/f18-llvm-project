! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LINE: func @_QPtest1
subroutine test1(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c(n)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[C]]
  ! CHECK: %[[rv:.*]] = fir.addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1

! CHECK-LINE: func @_QPtest2
subroutine test2(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c(:)
!  a = b + c
end subroutine test2

! CHECK-LINE: func @_QPtest3
subroutine test3(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK-DAG: %[[C:.*]] = fir.load %arg2
  ! CHECK: %[[rv:.*]] = fir.addf %[[Bi]], %[[C]]
  ! CHECK: %[[Ti:.*]] = fir.array_update %{{.*}}, %[[rv]], %
  ! CHECK: fir.result %[[Ti]]
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test3

! CHECK-LINE: func @_QPtest4
subroutine test4(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
  real :: a(100) ! FIXME: fake it for now
  real, intent(in) :: b(:), c
  ! CHECK: %[[Ba:.*]] = fir.box_addr %arg1
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %[[Ba]](%
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test4

! CHECK-LINE: func @_QPtest5
subroutine test5(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
!  real, pointer, intent(in) :: b(:)
  real :: a(100), b(100) ! FIXME: fake it for now
  real, intent(in) :: c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test5

! CHECK-LINE: func @_QPtest6
subroutine test6(a,b,c,n,m)
  integer :: n, m
  real, intent(out) :: a(n)
  real, intent(in) :: b(m), c
!  a(3:n:4) = b + c
end subroutine test6

! This is NOT a conflict. `a` appears on both the lhs and rhs here, but there
! are no loop-carried dependences and no copy is needed.
! CHECK-LINE: func @_QPtest7
subroutine test7(a,b,n)
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  ! CHECK: %[[Aout:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[Ain:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[Ain]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[B]]
  ! CHECK: %[[rv:.*]] = fir.addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = a + b
  ! CHECK: fir.array_merge_store %[[Aout]], %[[T]] to %arg0
end subroutine test7

! This FORALL construct does present a potential loop-carried dependence if
! implemented naively (and incorrectly). The final value of a(3) must be the
! value of a(2) before alistair begins execution added to b(2).
! CHECK-LINE: func @_QPtest8
subroutine test8(a,b,c,n,m)
  integer :: n, m
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  alistair: FORALL (i=1:n-1)
     a(i+1) = a(i) + b(i)
  END FORALL alistair
end subroutine test8
