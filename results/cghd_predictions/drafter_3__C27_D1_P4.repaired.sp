* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n4 n8 1u
C2 n5 n7 1u
R1 n4 n9 1k
R2 n4 n5 1k
R3 n4 0 1k
C3 n1 n4 1u
Q1 n3 n5 n6 QNPNdefault
R4 n2 n3 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n7 n7 0 1e+09
Rshunt_n8 n8 0 1e+09

.op
.end
