* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n9 n10 0 QNPNdefault
R1 0 n10 1k
R2 n1 n2 1k
Q2 n6 n1 0 QNPNdefault
Q3 n1 n4 n5 QNPNdefault
Q4 n5 n8 n9 QNPNdefault
R3 n3 n4 1k
R4 n7 n8 1k
R5 n2 n6 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end
