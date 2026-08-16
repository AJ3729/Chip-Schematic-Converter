* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n3 n4 DC 5
R1 n6 n5 1k
R2 n5 n7 1k
R3 n2 n6 1k
V2 0 n6 DC 5
C1 n3 n5 1u
Q1 n1 n1 n1 QNPNdefault
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
