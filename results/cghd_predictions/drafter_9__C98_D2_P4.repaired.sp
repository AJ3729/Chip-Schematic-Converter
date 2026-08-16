* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n4 n2 1k
V1 n4 0 DC 5
R2 n2 n5 1k
V2 n2 n3 DC 5
V3 n1 0 DC 5
R3 n1 n4 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
