* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n1 0 Ddefault
* SAME_NODE_SKIPPED I-DC both_on=0
* SAME_NODE_SKIPPED I-DC both_on=0
M1 0 0 0 0 PMOSdefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Resistor both_on=0
M2 n4 0 n4 n4 NMOSdefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED Diode both_on=0
* SAME_NODE_SKIPPED V-DC both_on=0
* SAME_NODE_SKIPPED Zener Diode both_on=0
M3 0 n6 n5 n5 NMOSdefault
M4 n2 n3 0 0 PMOSdefault
* SAME_NODE_SKIPPED Resistor both_on=0
* SAME_NODE_SKIPPED V-DC both_on=0
.model Ddefault D
.model NMOSdefault NMOS
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
