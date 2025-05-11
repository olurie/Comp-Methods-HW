program lu_example
  use toolbox
  implicit none

  real(8), dimension(4,4) :: A, L, U, LU, A_test
  real(8), dimension(4)   :: b, x, b_test
  integer :: i,j

  ! Define A
  A = reshape([ &
       1.0d0, 5.0d0, 2.0d0, 3.0d0, &
       1.0d0, 6.0d0, 8.0d0, 6.0d0, &
       1.0d0, 6.0d0,11.0d0, 2.0d0, &
       1.0d0, 7.0d0,17.0d0, 4.0d0 ], &
       shape(A), order=[2,1])

  ! Define b
  b = [1.0d0, 2.0d0, 1.0d0, 1.0d0]

 
    ! Print matrix A
    write(*,'(a)') 'A = '
    do j = 1, 4
        write(*,'(4f8.2)') A(j, :)
    end do

    ! Perform LU decomposition
    call lu_dec(A, L, U)

    ! Check the decomposition result
    A_test = matmul(L, U)

    ! Print L, U, and L*U
    write(*,'(/a)') 'L = '
    do j = 1, 4
        write(*,'(4f8.2)') L(j, :)
    end do

    write(*,'(/a)') 'U = '
    do j = 1, 4
        write(*,'(4f8.2)') U(j, :)
    end do

    write(*,'(/a)') 'A_test = L * U = '
    do j = 1, 4
        write(*,'(4f8.2)') A_test(j, :)
    end do

    write(*,'(/a/)') '-----------------------------'

    ! Solve Ax = b using lu_solve (x is overwritten with the solution)
    x = b
    call lu_solve(A, x)

    ! Verify the solution: A * x ≈ b
    b_test = matmul(A, x)

    ! Print solution and checks
    write(*,'(a)') 'x = '
    do j = 1, 4
        write(*,'(f8.2)') x(j)
    end do

    write(*,'(/a)') 'A * x = '
    do j = 1, 4
        write(*,'(f8.2)') b_test(j)
    end do

    write(*,'(/a)') 'Original b = '
    do j = 1, 4
        write(*,'(f8.2)') b(j)
    end do


end program lu_example
