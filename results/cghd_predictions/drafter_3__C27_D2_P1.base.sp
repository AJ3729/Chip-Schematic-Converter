* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n5 n7 1u
C2 n8 n4 1u
C3 n2 n3 1u
R1 n4 n5 1k
R2 n4 n8 1k
D1 n3 n4 Ddefault
Q1 n1 n5 n6 QNPNdefault
R3 n3 0 1k
.model Ddefault D
.model QNPNdefault NPN

.op
.end
