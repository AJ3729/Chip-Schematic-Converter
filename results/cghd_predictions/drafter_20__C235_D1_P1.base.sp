* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n8 n25 n8 QPNPdefault
* SAME_NODE_SKIPPED Diode both_on=n8
R1 n8 n11 1k
* SAME_NODE_SKIPPED Resistor both_on=n8
D1 n8 n2 Ddefault
* SAME_NODE_SKIPPED Resistor both_on=n8
Q2 n8 n29 n30 QNPNdefault
Q3 n2 n8 n9 QNPNdefault
D2 n8 n19 Ddefault
Q4 n8 n8 n8 QPNPdefault
* SAME_NODE_SKIPPED Diode both_on=n8
* SAME_NODE_SKIPPED Resistor both_on=n8
R2 n8 0 1k
* SAME_NODE_SKIPPED Resistor both_on=n8
R3 n1 n3 1k
Q5 n8 n25 n8 QPNPdefault
* UNSNAPPED Capacitor raw_nodes=[1, None]
R4 n35 n8 1k
R5 n9 n8 1k
R6 n1 n2 1k
C1 n23 n27 1u
* SAME_NODE_SKIPPED Resistor both_on=n8
R7 n1 n4 1k
R8 n8 n22 1k
R9 n8 n25 1k
R10 n6 n8 1k
R11 n28 n33 1k
R12 n1 n3 1k
R13 n15 n8 1k
R14 n15 n13 1k
* SAME_NODE_SKIPPED Resistor both_on=n8
* SAME_NODE_SKIPPED Resistor both_on=n8
* SAME_NODE_SKIPPED Resistor both_on=n8
R15 n31 n36 1k
R16 n13 n23 1k
D3 n14 n8 Ddefault
R17 n12 n13 1k
R18 n11 n8 1k
R19 n30 n31 1k
* SAME_NODE_SKIPPED Resistor both_on=n8
* SAME_NODE_SKIPPED Resistor both_on=n8
* SAME_NODE_SKIPPED Diode both_on=n8
Q6 n5 n3 n1 QPNPdefault
Q7 n8 n26 n8 QPNPdefault
C2 n13 n8 1u
C3 n23 n27 1u
* SAME_NODE_SKIPPED Capacitor both_on=n28
C4 n7 n5 1u
R20 n21 n28 1k
* SAME_NODE_SKIPPED Diode both_on=n8
* SAME_NODE_SKIPPED Resistor both_on=n8
R21 n8 n26 1k
Q8 n2 n8 n8 QNPNdefault
C5 n32 n35 1u
R22 n7 n10 1k
R23 n16 n8 1k
* SAME_NODE_SKIPPED Diode both_on=n8
* SAME_NODE_SKIPPED Capacitor both_on=n8
Q9 n14 n8 n8 QNPNdefault
R24 n8 n20 1k
R25 n18 n24 1k
* SAME_NODE_SKIPPED Resistor both_on=n13
R26 n18 n24 1k
R27 n13 n8 1k
Q10 n34 n31 n29 QPNPdefault
Q11 n2 n9 n8 QNPNdefault
C6 n36 n8 1u
Q12 n34 n31 n29 QNPNdefault
Q13 n3 n8 n5 QNPNdefault
* UNSNAPPED Capacitor raw_nodes=[1, None]
C7 n5 n8 1u
Q14 n11 n8 n2 QPNPdefault
R28 n12 n13 1k
R29 n24 n18 1k
V1 n17 0 DC 5
C8 n8 n29 1u
V2 n36 0 DC 5
R30 n33 n28 1k
R31 n16 n8 1k
* SAME_NODE_SKIPPED Capacitor both_on=n8
C9 n28 n29 1u
C10 n8 0 1u
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

.op
.end
