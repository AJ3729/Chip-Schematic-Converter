* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n2 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=n3
Q1 n2 n3 n2 QPNPdefault
Q2 0 n4 n5 QPNPdefault
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
Q3 n3 n3 n2 QPNPdefault
Q4 n2 n3 n3 QPNPdefault
Q5 n4 n1 n4 QPNPdefault
M1 n3 n3 n2 n2 PMOSdefault
Q6 n3 n2 n3 QPNPdefault
R2 n2 n1 1k
.model PMOSdefault PMOS
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
