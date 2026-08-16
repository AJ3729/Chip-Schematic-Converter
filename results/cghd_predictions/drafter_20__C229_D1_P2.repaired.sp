* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n3 0 Ddefault
D2 n5 0 Ddefault
R1 0 n6 1k
R2 0 n6 1k
R3 n1 0 1k
R4 0 n2 1k
R5 n1 0 1k
R6 n1 0 1k
Q1 0 0 n2 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=0
D3 n2 0 Zdefault
* UNSNAPPED MOSFET-P raw_nodes=[3, None, None]
* SAME_NODE_SKIPPED Zener Diode both_on=0
Q2 n4 n2 0 QPNPdefault
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09
Rshunt_n4 n4 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
