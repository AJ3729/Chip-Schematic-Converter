* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Diode both_on=n1
Q1 n1 n16 n18 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
Q2 n1 n1 n5 QNPNdefault
R1 n18 n11 1k
Q3 n12 n1 n12 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
Q4 n1 n1 n1 QNPNdefault
R2 n1 n9 1k
R3 n16 n18 1k
D1 n1 n11 Ddefault
* SAME_NODE_SKIPPED Diode both_on=n1
R4 n14 n16 1k
Q5 n1 n1 n1 QNPNdefault
R5 n5 n1 1k
* SAME_NODE_SKIPPED Diode both_on=n1
R6 n1 n13 1k
R7 n7 n1 1k
R8 n2 n1 1k
Q6 n1 n1 n11 QNPNdefault
R9 n2 n1 1k
Q7 n1 n1 n14 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
R10 n7 n1 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
Q8 n2 n1 n2 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
C1 n9 0 1u
Q9 n1 n1 n1 QPNPdefault
R11 n1 n18 1k
C2 n15 n11 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
Q10 n11 n10 n1 QPNPdefault
C3 n2 n3 1u
* SAME_NODE_SKIPPED Capacitor both_on=n1
R12 n1 n12 1k
R13 n2 n1 1k
Q11 n1 n1 n1 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
Q12 n11 n1 n1 QNPNdefault
R14 n1 n18 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
C4 n17 n18 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
R15 n4 n1 1k
Q13 n11 n1 n10 QPNPdefault
Q14 n1 n6 n1 QPNPdefault
R16 n5 n1 1k
R17 n1 n5 1k
R18 n1 n10 1k
R19 n11 n1 1k
* SAME_NODE_SKIPPED Capacitor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n11
Q15 n8 n6 n1 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
