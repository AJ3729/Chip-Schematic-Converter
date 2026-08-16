* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n4 0 AC 1m
L1 n3 n2 1m
C1 n2 0 1u
V1 n6 0 DC 5
C2 n2 0 1u
C3 n2 0 1u
C4 n1 0 1u
L2 n1 n4 1m
R1 n2 n5 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end
