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
!  real, allocatable, intent(out) :: a(:)
  real, intent(in) :: b(:), c
!  a = b + c
end subroutine test4

! CHECK-LINE: func @_QPtest5
subroutine test5(a,b,c)
!  real, allocatable, intent(out) :: a(:)
!  real, pointer, intent(in) :: b(:)
  real, intent(in) :: c
!  a = b + c
end subroutine test5

! CHECK-LINE: func @_QPtest6
subroutine test6(a,b,c,n,m)
  integer :: n, m
  real, intent(out) :: a(n)
  real, intent(in) :: b(m), c
!  a(3:n:4) = b + c
end subroutine test6
