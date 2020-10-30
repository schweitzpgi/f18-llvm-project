! RUN: bbc %s -o - | FileCheck %s

! CHECK-LINE: func @_QPtest1
subroutine test1(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c(n)
  a = b + c
end subroutine test1

! CHECK-LINE: func @_QPtest2
subroutine test2(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c(:)
!  a = b + c
end subroutine test2

! CHECK-LINE: func @_QPtest3
subroutine test3(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c
!  a = b + c
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
