* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n2 AC 1m
I2 n5 0 AC 1m
R1 n1 n3 1k
R2 n1 0 1k
R3 n3 n4 1k
I3 n4 n3 AC 1m
R4 n5 n6 1k
R5 n1 n2 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n5 n5 0 1e+09

.op
.end
