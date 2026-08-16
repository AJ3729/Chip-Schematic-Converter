* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n2 n2 n5 QNPNdefault
Q2 n2 n4 n2 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n2
Q3 n2 n2 n2 QPNPdefault
* SAME_NODE_SKIPPED Diode both_on=n2
R1 n2 n10 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
* SAME_NODE_SKIPPED Diode both_on=n2
* SAME_NODE_SKIPPED Diode both_on=n2
R2 n15 n2 1k
Q4 n13 n10 n2 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=n2
R3 n8 n2 1k
Q5 n2 n14 n15 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n2
R4 n2 n11 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
R5 n2 n12 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
* SAME_NODE_SKIPPED Resistor both_on=n2
R6 n2 n14 1k
Q6 n2 n2 n2 QPNPdefault
R7 n2 n9 1k
* SAME_NODE_SKIPPED Resistor both_on=n2
Q7 n2 n2 n2 QNPNdefault
D1 n7 n2 Ddefault
Q8 n2 n2 n2 QPNPdefault
Q9 n2 n2 n2 QPNPdefault
Q10 n13 n12 n13 QPNPdefault
R8 n8 n2 1k
R9 n12 n15 1k
R10 n5 n2 1k
R11 n12 n15 1k
Q11 n2 n2 n2 QPNPdefault
C1 n9 0 1u
R12 n5 n2 1k
Q12 n7 n2 n2 QNPNdefault
R13 n3 n2 1k
R14 n14 n15 1k
R15 n2 n5 1k
* SAME_NODE_SKIPPED Capacitor both_on=n2
* SAME_NODE_SKIPPED Diode both_on=n2
Q13 n2 n2 n2 QNPNdefault
Q14 n2 n2 n2 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n2
* SAME_NODE_SKIPPED Resistor both_on=n2
R16 n2 n12 1k
* SAME_NODE_SKIPPED Capacitor both_on=n2
R17 n4 n2 1k
* SAME_NODE_SKIPPED Capacitor both_on=n17
C2 n16 n15 1u
* SAME_NODE_SKIPPED Inductor both_on=n2
Q15 n7 n2 n2 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n2
* SAME_NODE_SKIPPED Capacitor both_on=n2
* SAME_NODE_SKIPPED Resistor both_on=n2
* SAME_NODE_SKIPPED Capacitor both_on=n11
* SAME_NODE_SKIPPED Resistor both_on=n2
C3 n2 n1 1u
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
