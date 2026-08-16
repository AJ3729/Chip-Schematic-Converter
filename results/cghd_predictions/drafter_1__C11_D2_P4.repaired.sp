* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n2 0 1u
I1 n3 0 AC 1m
R1 n2 n3 1k
C2 0 n2 1u
L1 n1 0 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end
