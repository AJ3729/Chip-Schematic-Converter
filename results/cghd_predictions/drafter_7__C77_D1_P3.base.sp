* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n4 0 0 QNPNdefault
R1 n1 n2 1k
R2 0 n6 1k
Q2 n6 0 n5 QNPNdefault
R3 n1 0 1k
D1 n4 n5 Ddefault
R4 n1 0 1k
R5 n1 n3 1k
D2 n7 n2 Ddefault
Q3 0 n2 n3 QNPNdefault
D3 n7 n2 Ddefault
V1 n2 0 DC 5
.model Ddefault D
.model QNPNdefault NPN

.op
.end
