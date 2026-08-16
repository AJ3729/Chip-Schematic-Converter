* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n1 n15 n16 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n1
R1 n1 n8 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
D1 n6 n1 Ddefault
R2 n1 n10 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Diode both_on=n1
D2 n11 n1 Ddefault
R3 n16 n11 1k
R4 n1 n16 1k
Q2 n13 n12 n1 QPNPdefault
Q3 n13 n1 n13 QPNPdefault
R5 n7 n1 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
R6 n15 n16 1k
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
R7 n14 n15 1k
Q4 n11 n1 n9 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=n1
R8 n7 n1 1k
* SAME_NODE_SKIPPED Diode both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n11
C1 n8 0 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
* SAME_NODE_SKIPPED Resistor both_on=n1
R9 n5 n1 1k
R10 n1 n16 1k
* SAME_NODE_SKIPPED Capacitor both_on=n1
* SAME_NODE_SKIPPED Capacitor both_on=n1
R11 n1 n9 1k
* SAME_NODE_SKIPPED Diode both_on=n1
Q5 n1 n1 n1 QPNPdefault
Q6 n6 n1 n1 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n1
* SAME_NODE_SKIPPED Diode both_on=n1
R12 n3 n1 1k
* UNSNAPPED Capacitor raw_nodes=[None, None]
Q7 n11 n1 n1 QNPNdefault
R13 n1 n5 1k
Q8 n1 n1 n1 QPNPdefault
Q9 n1 n1 n1 QNPNdefault
C2 n1 n4 1u
* SAME_NODE_SKIPPED Resistor both_on=n1
C3 n1 n2 1u
Q10 n11 n9 n1 QPNPdefault
Q11 n1 n1 n14 QNPNdefault
Q12 n11 n1 n1 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n11
* SAME_NODE_SKIPPED Capacitor both_on=n16
Q13 n11 n9 n1 QPNPdefault
Q14 n1 n1 n1 QNPNdefault
Q15 n1 n5 n1 QPNPdefault
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
