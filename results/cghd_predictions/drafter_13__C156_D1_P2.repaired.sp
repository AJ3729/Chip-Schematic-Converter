* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n4 0 DC 5
I1 n4 n3 AC 1m
R1 n2 n3 1k
R2 n5 n2 1k
M1 n1 n1 n1 n1 PMOSdefault
V2 n5 n3 DC 5
I2 0 n2 DC 1m
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end
