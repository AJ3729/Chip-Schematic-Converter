* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 n4 n2 0 0 PMOSdefault
* UNSNAPPED MOSFET-P raw_nodes=[3, 2, None]
* SAME_NODE_SKIPPED Zener Diode both_on=0
E1 0 0 0 0 100k
* UNSNAPPED MOSFET-N raw_nodes=[2, 48, None]
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
