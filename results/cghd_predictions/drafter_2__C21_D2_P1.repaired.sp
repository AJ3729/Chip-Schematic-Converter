* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 0 n1 DC 1m
R1 0 n2 1k
R2 0 n9 1k
C1 n3 0 1u
* SAME_NODE_SKIPPED Capacitor both_on=0
V1 n8 0 DC 5
I2 n5 0 DC 1m
M1 0 n4 n7 n7 PMOSdefault
M2 0 n4 n7 n7 PMOSdefault
E1 n6 0 n3 0 100k
* SAME_NODE_SKIPPED Resistor both_on=0
V2 n8 0 DC 5
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end
