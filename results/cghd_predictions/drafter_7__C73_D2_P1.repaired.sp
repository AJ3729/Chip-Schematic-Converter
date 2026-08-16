* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n3 n4 QNPNdefault
R1 n10 n11 1k
R2 0 n1 1k
Q2 n5 0 n9 QNPNdefault
Q3 n8 n11 n9 QNPNdefault
R3 n2 n3 1k
R4 n6 n7 1k
Q4 n4 n7 n8 QNPNdefault
R5 n1 n5 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end
