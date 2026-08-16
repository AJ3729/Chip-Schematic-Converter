* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 n1 n3 1u
R1 0 n1 1k
I1 0 n2 DC 1m
I2 0 n2 AC 1m
V1 0 n2 AC 1

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end
