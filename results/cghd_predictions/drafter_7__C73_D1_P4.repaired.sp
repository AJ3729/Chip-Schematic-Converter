* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n6 n5 1k
R2 n3 n2 1k
R3 0 n1 1k
Q1 n8 n9 n7 QNPNdefault
Q2 n4 n5 n8 QNPNdefault
R4 n10 n9 1k
Q3 0 n2 n4 QNPNdefault
Q4 n1 0 n7 QNPNdefault
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
