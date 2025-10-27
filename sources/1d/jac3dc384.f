        PROGRAM    JAC3D
        PARAMETER    (L=384,  ITMAX=100)
        REAL  A(L,L,L),  B(L,L,L)
        REAL EPS, MAXEPS
!DVM$   DISTRIBUTE     ( BLOCK,   BLOCK, *)   ::   A
!DVM$   ALIGN  B(I,J,K)  WITH  A(I,J,K)
C		 arrays A and B  with block distribution 

        PRINT *,  '**********  TEST_JAC3D   **********'
                  MAXEPS  =  0.5
!DVM$   REGION
!DVM$   PARALLEL    (K,J,I)   ON   A(I,J,K)
C		nest of two parallel loops, iteration (i,j,k) will be executed on 
C		processor, which is owner of element A(i,j,k)
            DO    K  =  1, L 
            DO    J  =  1, L
            DO    I  =  1, L
                A(I,J,K)  =  0.
                IF(I.EQ.1 .OR. J.EQ.1 .OR. I.EQ.L 
     &                    .OR. J.EQ.L .OR. K.EQ.1 .OR. K.EQ.L) THEN
                      B(I,J,K) = 0.
                ELSE
                      B(I,J,K) = ( 1 + I + J + K)
                ENDIF
            ENDDO
            ENDDO
            ENDDO 
!DVM$   END REGION
        DO    IT  =  1,  ITMAX
                  EPS  =  0.
!DVM$   ACTUAL(EPS)
                  DO   K  =  2, L-1
!DVM$   REGION
!DVM$   PARALLEL  (J,I)   ON  A(I,J,*),  REDUCTION ( MAX( EPS ))
C		variable EPS is used for calculation of maximum value

                  DO   J  =  2, L-1
                  DO   I  =  2, L-1
                         EPS = MAX ( EPS, ABS (B( I,J,K) - A( I,J,K)))
                         A(I,J,K)  =  B(I,J,K)
                  ENDDO
                  ENDDO
!DVM$   END REGION
                  ENDDO
                  DO   K = 2,  L-1
!DVM$   REGION
!DVM$   PARALLEL  (J,I)   ON  B(I,J,*),   SHADOW_RENEW (A)
C		Copying shadow elements of array A from 
C		neighbouring processors before loop execution

                  DO   J = 2,  L-1
                  DO   I = 2,  L-1
        B(I,J,K) =  (A( I-1,J,K) + A( I,J-1,K ) + A( I,J,K-1)+
     *               A( I+1,J,K) + A( I,J+1,K ) + A( I,J,K+1)) / 6.0
                  ENDDO
                  ENDDO
!DVM$   END REGION
                  ENDDO
!DVM$   GET_ACTUAL(EPS)
                  PRINT 200,  IT, EPS
200               FORMAT(' IT = ',I4, '   EPS = ', E14.7)
                  IF ( EPS . LT . MAXEPS )    EXIT
        ENDDO
!!DVM$   GET_ACTUAL(B)    
!        OPEN (3, FILE='JAC.DAT', FORM='FORMATTED', STATUS='UNKNOWN')
!        WRITE (3,*)   B
!        CLOSE (3)
        END

