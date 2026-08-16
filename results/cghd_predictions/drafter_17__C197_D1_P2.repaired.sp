* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n2 1u
R1 n4 n2 1k
C2 0 n4 1u
R2 n3 n2 1k
C3 n3 n2 1u
R3 0 n3 1k
I1 n3 0 AC 1m
C4 0 n2 1u
I2 0 n4 AC 1m
L1 n1 n5 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
