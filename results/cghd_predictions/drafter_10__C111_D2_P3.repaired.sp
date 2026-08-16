* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n3 0 AC 1
R1 n3 n4 1k
R2 n5 0 1k
C1 n4 n5 1u
V2 n2 n1 DC 5
L1 n5 0 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
