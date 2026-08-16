* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n11 n1 Ddefault
D2 n22 n5 Ddefault
Q1 n5 n22 n27 QPNPdefault
R1 n8 n5 1k
Q2 n22 n22 n30 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n8
Q3 n1 n5 n10 QNPNdefault
R2 n22 n5 1k
D3 n5 n11 Ddefault
D4 n14 n5 Ddefault
Q4 n5 n7 n5 QPNPdefault
Q5 n5 n27 n5 QPNPdefault
R3 n8 n20 1k
* SAME_NODE_SKIPPED Diode both_on=n22
R4 n2 n4 1k
R5 n8 n12 1k
Q6 n5 n4 n2 QPNPdefault
R6 n15 n5 1k
R7 n10 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
R8 n21 n17 1k
R9 n2 n1 1k
R10 n5 n27 1k
R11 n22 0 1k
R12 n32 n5 1k
R13 n19 n5 1k
R14 n20 n5 1k
R15 n12 n19 1k
R16 n4 n5 1k
R17 n11 n24 1k
R18 n13 n25 1k
R19 n15 n13 1k
R20 n22 n34 1k
R21 n2 n5 1k
R22 n30 n31 1k
R23 n6 n5 1k
R24 n16 n5 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
R25 n17 n26 1k
R26 n13 n22 1k
R27 n2 n5 1k
R28 n22 n33 1k
R29 n31 n35 1k
R30 n19 n11 1k
Q7 n22 n5 n5 QPNPdefault
* SAME_NODE_SKIPPED Diode both_on=n5
Q8 n1 n9 n8 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n32
C1 n17 n23 1u
Q9 n22 n31 n35 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n13
* SAME_NODE_SKIPPED Capacitor both_on=n5
C2 n25 n29 1u
C3 n13 n22 1u
R31 n16 n22 1k
R32 n13 n26 1k
* SAME_NODE_SKIPPED Capacitor both_on=n5
R33 n13 n22 1k
* SAME_NODE_SKIPPED Capacitor both_on=n5
Q10 n1 n8 n12 QNPNdefault
* SAME_NODE_SKIPPED Diode both_on=n5
* SAME_NODE_SKIPPED Resistor both_on=n5
Q11 n5 n5 n5 QPNPdefault
Q12 n1 n10 n8 QNPNdefault
V1 n18 0 DC 5
* SAME_NODE_SKIPPED Resistor both_on=n5
* SAME_NODE_SKIPPED Diode both_on=n5
R34 n13 n22 1k
* SAME_NODE_SKIPPED Resistor both_on=n5
* SAME_NODE_SKIPPED Capacitor both_on=n22
R35 n5 n13 1k
Q13 n22 n16 n5 QPNPdefault
C4 n25 n29 1u
* SAME_NODE_SKIPPED Capacitor both_on=n22
R36 n7 n17 1k
C5 n36 n5 1u
C6 n5 n22 1u
R37 n5 n13 1k
C7 n1 n3 1u
R38 n9 n8 1k
Q14 n26 n22 n26 QPNPdefault
Q15 n14 n5 n5 QNPNdefault
* SAME_NODE_SKIPPED Capacitor both_on=n5
R39 n7 n13 1k
R40 n5 n13 1k
R41 n13 n22 1k
C8 n28 n24 1u
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
