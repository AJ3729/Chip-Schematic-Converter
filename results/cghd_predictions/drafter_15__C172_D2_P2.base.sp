* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED I-AC both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
M1 n1 n4 n5 n5 NMOSdefault
C1 n1 n3 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
C2 n2 n1 1u
I1 n6 n1 AC 1m
* SAME_NODE_SKIPPED Capacitor both_on=n1
I2 n6 n1 DC 1m
M2 n1 n1 n1 n1 NMOSdefault
M3 n1 n1 n1 n1 PMOSdefault
.model NMOSdefault NMOS
.model PMOSdefault PMOS

.op
.end
