* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n2 n1 DC 1m
C1 n2 n3 1u
R1 n2 n1 1k
C2 0 n4 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end
