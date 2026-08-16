* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n2 0 1u
R1 n3 n2 1k
C2 n2 n1 1u
V1 n4 0 AC 1
R2 n1 0 1k
V2 n4 0 AC 1

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
