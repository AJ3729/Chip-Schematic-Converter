* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n3 1k
V1 n4 n3 DC 5
R2 n1 n2 1k
R3 n1 0 1k
R4 n5 n6 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
