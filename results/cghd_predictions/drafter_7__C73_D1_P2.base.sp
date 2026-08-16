* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n3 n5 QNPNdefault
R1 0 n2 1k
R2 n8 n9 1k
Q2 n6 n9 n7 QNPNdefault
Q3 n4 0 n7 QNPNdefault
* UNSNAPPED BJT-NPN raw_nodes=[5, 6, None]
V1 n1 0 DC 5
.model QNPNdefault NPN

.op
.end
