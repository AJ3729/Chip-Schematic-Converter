* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n2 0 1k
R2 0 n5 1k
R3 0 n5 1k
R4 n1 n3 1k
Q1 n3 0 0 QNPNdefault
Q2 0 n4 n2 QPNPdefault
R5 n1 n2 1k
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
