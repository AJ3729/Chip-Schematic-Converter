* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n5 n4 1k
C1 0 n2 1u
* UNSNAPPED Resistor raw_nodes=[4, None]
Q1 n3 n3 n3 QPNPdefault
M1 n2 n1 0 0 PMOSdefault
* UNSNAPPED Inductor raw_nodes=[2, None]
.model PMOSdefault PMOS
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n6 n6 0 1e+09
Rshunt_n7 n7 0 1e+09

.op
.end
