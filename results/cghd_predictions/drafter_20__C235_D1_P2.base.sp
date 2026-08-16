* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n5 n22 n5 QPNPdefault
Q2 n1 n5 n10 QNPNdefault
Q3 n5 n5 n22 QPNPdefault
Q4 n1 n12 n15 QNPNdefault
Q5 n1 n12 n12 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n5
Q6 n1 n10 n12 QNPNdefault
Q7 n5 n18 n23 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n5
Q8 n5 n5 n5 QNPNdefault
D1 n14 n1 Ddefault
Q9 n5 n5 n5 QPNPdefault
R1 n5 0 1k
R2 n15 n17 1k
Q10 n11 n5 n5 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n5
D2 n5 n19 Ddefault
R3 n8 n5 1k
R4 n14 n19 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
Q11 n5 n2 n2 QPNPdefault
Q12 n18 n26 n27 QNPNdefault
R5 n29 n5 1k
R6 n17 n21 1k
R7 n5 n22 1k
C1 n2 n5 1u
R8 n5 n18 1k
R9 n1 n2 1k
R10 n12 n15 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
R11 n18 n25 1k
R12 n12 n5 1k
R13 n8 n5 1k
R14 n10 n5 1k
R15 n4 n5 1k
R16 n17 n14 1k
R17 n23 n26 1k
R18 n2 n5 1k
C2 n28 n29 1u
R19 n5 n21 1k
D3 n11 n5 Ddefault
Q13 n2 n3 n1 QPNPdefault
R20 n1 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
R21 n6 n9 1k
R22 n12 n5 1k
R23 n1 n2 1k
R24 n5 n24 1k
R25 n26 n27 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
* SAME_NODE_SKIPPED Resistor both_on=n5
* SAME_NODE_SKIPPED Diode both_on=n5
C3 n5 n19 1u
Q14 n5 n24 n21 QPNPdefault
R26 n2 n3 1k
R27 n5 n18 1k
R28 n2 n5 1k
V1 n13 0 DC 5
C4 n16 n20 1u
R29 n9 n5 1k
R30 n7 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
C5 n16 n20 1u
Q15 n2 n6 n7 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n1
Q16 n5 n18 n18 QNPNdefault
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
