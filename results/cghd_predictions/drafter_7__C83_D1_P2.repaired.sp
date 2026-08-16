* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n1 1k
R2 0 n2 1k
R3 0 n3 1k
I1 n2 n1 DC 1m
R4 n3 n5 1k
R5 n7 n8 1k
I2 n7 n2 DC 1m
L1 n6 n5 1m
I3 n5 n4 DC 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n4 n4 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end
