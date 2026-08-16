* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n2 1k
R2 n1 n3 1k
R3 n6 n7 1k
R4 n2 n5 1k
Q1 n5 0 n9 QNPNdefault
Q2 0 n3 n4 QNPNdefault
R5 n10 n11 1k
Q3 n4 n7 n8 QNPNdefault
Q4 n8 n11 n9 QNPNdefault
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n10 n10 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end
