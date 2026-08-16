* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 0 0 0 0 100k
I1 n9 0 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=0
E2 0 0 n7 0 100k
R1 0 n5 1k
C1 n5 n1 1u
R2 n3 0 1k
R3 0 n6 1k
C2 n8 0 1u
C3 0 n2 1u
R4 0 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=0
R5 n1 0 1k
R6 0 n9 1k
R7 0 n8 1k
I2 0 n4 AC 1m
V1 0 n4 DC 5
R8 n6 0 1k
L1 n7 0 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
