* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n4 n5 0 QNPNdefault
D1 0 n2 Ddefault
Q2 n4 n3 n4 QNPNdefault
R1 n1 n3 1k
R2 n5 0 1k
R3 n1 n3 1k
Q3 n3 n2 n5 QNPNdefault
R4 n1 n2 1k
D2 0 n2 Ddefault
.model Ddefault D
.model QNPNdefault NPN

.op
.end
