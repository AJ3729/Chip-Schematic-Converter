* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n2 1k
R2 0 n3 1k
C1 n2 0 1u
R3 0 n3 1k
C2 n1 n3 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
