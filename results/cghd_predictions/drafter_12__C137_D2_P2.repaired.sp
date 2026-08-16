* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n2 1k
C1 n2 0 1u
R2 0 n3 1k
R3 0 n4 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09

.op
.end
