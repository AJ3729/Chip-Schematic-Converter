* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n2 0 1u
I1 n3 0 DC 1m
R1 n2 0 1k
C2 n1 n2 1u
I2 0 n4 AC 1m
I3 0 n1 DC 1m
L1 n1 0 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end
