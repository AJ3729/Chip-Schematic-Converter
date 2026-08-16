* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n2 0 Ddefault
Q1 0 n1 n3 QPNPdefault
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
M1 n1 0 n3 n3 PMOSdefault
.model Ddefault D
.model PMOSdefault PMOS
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
