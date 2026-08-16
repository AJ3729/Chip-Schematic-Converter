* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n3 0 n6 QNPNdefault
Q2 n4 n2 n3 QNPNdefault
R1 n2 n1 1k
R2 n1 0 1k
R3 0 n6 1k
R4 n1 n2 1k
R5 n1 0 1k
V1 n5 0 DC 5
D1 n6 0 Ddefault
M1 0 n4 n2 n2 PMOSdefault
* UNSNAPPED BJT-PNP raw_nodes=[1, 5, None]
Q3 0 n1 n2 QNPNdefault
.model Ddefault D
.model PMOSdefault PMOS
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n5 n5 0 1e+09

.op
.end
