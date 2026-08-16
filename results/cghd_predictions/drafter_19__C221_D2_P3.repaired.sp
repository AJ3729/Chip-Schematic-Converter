* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n5 1k
C1 n1 0 1u
* SAME_NODE_SKIPPED Resistor both_on=0
M1 0 n2 n1 n1 PMOSdefault
M2 n1 n3 n2 n2 PMOSdefault
Q1 0 n4 0 QPNPdefault
M3 n1 n2 n3 n3 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
.model PMOSdefault PMOS
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
