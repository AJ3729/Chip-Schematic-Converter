* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n1 n1 n4 QNPNdefault
Q2 n1 n15 n17 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n1
Q3 n13 n1 n11 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
R1 n17 n1 1k
* SAME_NODE_SKIPPED Diode both_on=n1
R2 n1 n7 1k
Q4 n1 n1 n12 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
R3 n6 n1 1k
R4 n1 n9 1k
R5 n12 n15 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
R6 n4 n1 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Diode both_on=n1
R7 n6 n1 1k
R8 n15 n17 1k
* SAME_NODE_SKIPPED Diode both_on=n1
Q5 n1 n1 n8 QPNPdefault
Q6 n1 n1 n1 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n1
R9 n1 n17 1k
* SAME_NODE_SKIPPED Diode both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
Q7 n1 n8 n1 QPNPdefault
Q8 n1 n1 n1 QNPNdefault
R10 n13 n17 1k
R11 n3 n1 1k
Q9 n1 n1 n1 QPNPdefault
C1 n16 n17 1u
C2 n16 n17 1u
Q10 n11 n10 n1 QPNPdefault
C3 n7 0 1u
R12 n1 n4 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
Q11 n1 n1 n1 QNPNdefault
M1 n1 n2 n1 n1 PMOSdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
R13 n1 n8 1k
D1 n5 n1 Ddefault
C4 n14 n1 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n1
C5 n1 n2 1u
Q12 n5 n1 n1 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
.model Ddefault D
.model PMOSdefault PMOS
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
