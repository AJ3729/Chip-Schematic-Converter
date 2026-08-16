* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n3 n4 1k
R2 n2 0 1k
R3 0 n1 1k
E1 n1 0 n4 0 100k

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
