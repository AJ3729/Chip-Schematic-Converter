* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n10 0 n8 n4 100k
R1 n4 n9 1k
R2 n8 n4 1k
R3 n9 n2 1k
R4 n6 n2 1k
R5 n1 n2 1k
R6 n4 0 1k
R7 n7 n2 1k
C1 n1 n3 1u
R8 n7 n4 1k
C2 n3 n6 1u
R9 n8 0 1k
R10 n10 0 1k
E2 n2 0 n9 n4 100k
C3 n4 0 1u
C4 n7 n9 1u
C5 n4 n5 1u
C6 n6 n7 1u
R11 n3 n2 1k
V1 n4 0 DC 5
C7 n2 n11 1u
D1 n4 n5 Ddefault
.model Ddefault D

.op
.end
