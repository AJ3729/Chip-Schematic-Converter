* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n4 0 1k
R2 n3 n1 1k
R3 n1 n2 1k
R4 0 n5 1k
E1 n2 0 0 n1 100k

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
