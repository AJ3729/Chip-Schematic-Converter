* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n2 n4 AC 1m
R1 n2 0 1k
R2 n3 n5 1k
R3 n3 n6 1k
R4 n3 0 1k
R5 n1 0 1k
C1 n1 0 1u
C2 0 n3 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09

.op
.end
