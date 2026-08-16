* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 0 n1 1k
Q1 n8 n10 n9 QNPNdefault
Q2 0 n3 n4 QNPNdefault
Q3 n5 0 n9 QNPNdefault
* UNSNAPPED Resistor raw_nodes=[9, None]
R2 n2 n3 1k
* UNSNAPPED Resistor raw_nodes=[6, None]
R3 n1 n5 1k
Q4 n4 n7 n8 QNPNdefault
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n10 n10 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end
